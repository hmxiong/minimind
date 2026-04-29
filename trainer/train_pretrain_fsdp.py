import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import time
import warnings
import torch
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim
from torch.utils.data import DataLoader, DistributedSampler
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy, StateDictType, FullStateDictConfig
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM, MiniMindBlock
from dataset.lm_dataset import PretrainDataset
from trainer.trainer_utils import get_lr, Logger, is_main_process, init_distributed_mode, setup_seed, SkipBatchSampler, get_model_params

warnings.filterwarnings('ignore')


def _parse_sharding_strategy(name: str) -> ShardingStrategy:
    name = (name or "full").lower()
    if name in {"full", "full_shard"}:
        return ShardingStrategy.FULL_SHARD
    if name in {"grad", "shard_grad_op"}:
        return ShardingStrategy.SHARD_GRAD_OP
    if name in {"hybrid", "hybrid_shard"}:
        return ShardingStrategy.HYBRID_SHARD
    if name in {"none", "no_shard"}:
        return ShardingStrategy.NO_SHARD
    raise ValueError(f"Unknown sharding strategy: {name}")


def _build_model_and_tokenizer(lm_config: MiniMindConfig):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    model = MiniMindForCausalLM(lm_config)

    if args.from_weight != 'none':
        moe_suffix = '_moe' if lm_config.use_moe else ''
        weight_path = f'{args.save_dir}/{args.from_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
        weights = torch.load(weight_path, map_location='cpu')
        model.load_state_dict(weights, strict=False)

    if is_main_process():
        get_model_params(model, lm_config)
        Logger(f'Trainable Params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f}M')

    return model, tokenizer


def _save_fsdp_checkpoint(epoch: int, step: int, lm_config: MiniMindConfig, model: FSDP, optimizer, scaler, wandb=None):
    moe_suffix = '_moe' if lm_config.use_moe else ''
    rank = dist.get_rank() if dist.is_initialized() else 0

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    weight_path = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
    resume_path = f'{args.checkpoint_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}_fsdp_rank{rank}.pth'

    dist.barrier()
    if is_main_process():
        model.eval()
        with FSDP.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
        ):
            state_dict = model.state_dict()
        state_dict = {k: v.half() for k, v in state_dict.items()}
        tmp = weight_path + ".tmp"
        torch.save(state_dict, tmp)
        os.replace(tmp, weight_path)
        del state_dict
        model.train()

    resume_data = {
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
        "step": step,
        "world_size": dist.get_world_size() if dist.is_initialized() else 1,
        "wandb_id": getattr(getattr(wandb, "get_run", lambda: None)(), "id", None) if wandb else None,
    }
    tmp = resume_path + ".tmp"
    torch.save(resume_data, tmp)
    os.replace(tmp, resume_path)

    dist.barrier()


def _try_resume_fsdp(lm_config: MiniMindConfig, model: FSDP, optimizer, scaler):
    moe_suffix = '_moe' if lm_config.use_moe else ''
    rank = dist.get_rank() if dist.is_initialized() else 0

    weight_path = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
    resume_path = f'{args.checkpoint_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}_fsdp_rank{rank}.pth'

    if not (os.path.exists(weight_path) and os.path.exists(resume_path)):
        return None

    with FSDP.state_dict_type(
        model,
        StateDictType.FULL_STATE_DICT,
        FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
    ):
        weights = torch.load(weight_path, map_location='cpu')
        model.load_state_dict(weights, strict=False)

    resume_data = torch.load(resume_path, map_location='cpu')
    optimizer.load_state_dict(resume_data["optimizer"])
    if scaler is not None and resume_data.get("scaler") is not None:
        scaler.load_state_dict(resume_data["scaler"])

    dist.barrier()
    return resume_data


def train_epoch(epoch, loader, iters, start_step=0, wandb=None):
    start_time = time.time()
    last_step = start_step

    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
        input_ids = input_ids.to(args.device, non_blocking=True)
        labels = labels.to(args.device, non_blocking=True)
        last_step = step

        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        need_sync = (step % args.accumulation_steps == 0)
        sync_ctx = nullcontext() if need_sync else model.no_sync()

        with sync_ctx:
            with autocast_ctx:
                res = model(input_ids, labels=labels)
                loss = res.loss + res.aux_loss
                loss = loss / args.accumulation_steps

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

        if need_sync:
            if scaler is not None:
                scaler.unscale_(optimizer)
            model.clip_grad_norm_(args.grad_clip)
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0 or step == iters:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps
            current_aux_loss = res.aux_loss.item() if res.aux_loss is not None else 0.0
            current_logits_loss = current_loss - current_aux_loss
            current_lr = optimizer.param_groups[-1]['lr']
            eta_min = spend_time / max(step - start_step, 1) * (iters - step) // 60
            Logger(
                f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), '
                f'loss: {current_loss:.4f}, logits_loss: {current_logits_loss:.4f}, aux_loss: {current_aux_loss:.4f}, '
                f'lr: {current_lr:.8f}, epoch_time: {eta_min:.1f}min'
            )
            if wandb:
                wandb.log(
                    {
                        "loss": current_loss,
                        "logits_loss": current_logits_loss,
                        "aux_loss": current_aux_loss,
                        "learning_rate": current_lr,
                        "epoch_time": eta_min,
                    }
                )

        if (step % args.save_interval == 0 or step == iters):
            _save_fsdp_checkpoint(epoch, step, lm_config, model, optimizer, scaler, wandb=wandb)

        del input_ids, labels, res, loss

    if last_step > start_step and last_step % args.accumulation_steps != 0:
        if scaler is not None:
            scaler.unscale_(optimizer)
        model.clip_grad_norm_(args.grad_clip)
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind Pretraining (FSDP)")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument("--checkpoint_dir", type=str, default="../checkpoints", help="训练状态保存目录")
    parser.add_argument("--save_weight", default="pretrain", type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=2, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=8, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=1000, help="模型保存间隔")
    parser.add_argument("--hidden_size", default=768, type=int, help="隐藏层维度")
    parser.add_argument("--num_hidden_layers", default=8, type=int, help="隐藏层数量")
    parser.add_argument("--max_seq_len", default=340, type=int, help="训练的最大截断长度（中文1token≈1.5~1.7字符）")
    parser.add_argument("--use_moe", default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument("--data_path", type=str, default="../dataset/pretrain_t2t_mini.jsonl", help="预训练数据路径")
    parser.add_argument("--from_weight", default="none", type=str, help="基于哪个权重训练，为none则从头开始")
    parser.add_argument("--from_resume", default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument("--tokenizer_path", type=str, default="../model", help="tokenizer目录")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Pretrain", help="wandb项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是）")
    parser.add_argument("--sharding", type=str, default="full", help="FSDP分片策略：full|grad|hybrid|none")
    args = parser.parse_args()

    local_rank = init_distributed_mode()
    if not dist.is_initialized():
        raise RuntimeError("FSDP训练需要使用 torchrun 启动并初始化分布式进程组")
    args.device = f"cuda:{local_rank}"
    setup_seed(42 + dist.get_rank())

    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)

    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_run_name = f"MiniMind-Pretrain-FSDP-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name)

    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe))
    base_model, tokenizer = _build_model_and_tokenizer(lm_config)

    if args.use_compile == 1:
        base_model = torch.compile(base_model)
        Logger("torch.compile enabled")

    sharding_strategy = _parse_sharding_strategy(args.sharding)
    mp = MixedPrecision(param_dtype=dtype, reduce_dtype=dtype, buffer_dtype=dtype) if device_type == "cuda" else None
    auto_wrap_policy = transformer_auto_wrap_policy(transformer_layer_cls={MiniMindBlock})

    model = FSDP(
        base_model,
        auto_wrap_policy=auto_wrap_policy,
        sharding_strategy=sharding_strategy,
        mixed_precision=mp,
        device_id=torch.cuda.current_device(),
        sync_module_states=True,
        use_orig_params=True,
    )

    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None

    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == "float16"))
    scaler = scaler if scaler.is_enabled() else None
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    start_epoch, start_step = 0, 0
    if args.from_resume == 1:
        resume_data = _try_resume_fsdp(lm_config, model, optimizer, scaler)
        if resume_data:
            start_epoch = resume_data.get("epoch", 0)
            start_step = resume_data.get("step", 0)
            if is_main_process():
                Logger(f"resume: epoch={start_epoch}, step={start_step}")

    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch + dist.get_rank())
        indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        if skip > 0 and is_main_process():
            Logger(f"Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始")
        train_epoch(epoch, loader, len(loader) + skip if skip > 0 else len(loader), start_step if skip > 0 else 0, wandb)

    dist.barrier()
    dist.destroy_process_group()
