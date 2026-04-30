import argparse
import math
import time
from typing import Callable, Dict, Optional

import torch
import torch.nn.functional as F


def parse_dtype(name: str) -> torch.dtype:
    n = name.lower()
    if n in {"fp16", "float16", "half"}:
        return torch.float16
    if n in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if n in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def make_inputs(
    bsz: int,
    n_heads: int,
    seq_len: int,
    head_dim: int,
    dtype: torch.dtype,
    device: str,
    requires_grad: bool,
):
    q = torch.randn(bsz, n_heads, seq_len, head_dim, device=device, dtype=dtype, requires_grad=requires_grad)
    k = torch.randn(bsz, n_heads, seq_len, head_dim, device=device, dtype=dtype, requires_grad=requires_grad)
    v = torch.randn(bsz, n_heads, seq_len, head_dim, device=device, dtype=dtype, requires_grad=requires_grad)
    return q, k, v


def attention_naive(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = True,
    dropout_p: float = 0.0,
    training: bool = False,
) -> torch.Tensor:
    scale = 1.0 / math.sqrt(q.size(-1))
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    if is_causal:
        s = q.size(-2)
        causal_mask = torch.ones((s, s), device=q.device, dtype=torch.bool).triu(1)
        scores = scores.masked_fill(causal_mask, float("-inf"))
    probs = torch.softmax(scores.float(), dim=-1).to(q.dtype)
    if dropout_p > 0 and training:
        probs = F.dropout(probs, p=dropout_p, training=True)
    return torch.matmul(probs, v)


def attention_sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = True,
    dropout_p: float = 0.0,
    training: bool = False,
) -> torch.Tensor:
    return F.scaled_dot_product_attention(
        q, k, v, is_causal=is_causal, dropout_p=dropout_p if training else 0.0
    )


def _maybe_sync(device: str):
    if "cuda" in device:
        torch.cuda.synchronize(device=device)


def benchmark_impl(
    name: str,
    fn: Callable[..., torch.Tensor],
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    warmup: int,
    iters: int,
    is_causal: bool,
    dropout_p: float,
    backward: bool,
) -> Dict[str, float]:
    device = str(q.device)
    if "cuda" in device:
        torch.cuda.reset_peak_memory_stats(device=q.device)

    # warmup
    for _ in range(warmup):
        out = fn(q, k, v, is_causal=is_causal, dropout_p=dropout_p, training=backward)
        if backward:
            loss = out.float().mean()
            loss.backward()
            q.grad = None
            k.grad = None
            v.grad = None
    _maybe_sync(device)

    start = time.perf_counter()
    for _ in range(iters):
        out = fn(q, k, v, is_causal=is_causal, dropout_p=dropout_p, training=backward)
        if backward:
            loss = out.float().mean()
            loss.backward()
            q.grad = None
            k.grad = None
            v.grad = None
    _maybe_sync(device)
    elapsed = time.perf_counter() - start

    peak_mem_mb = 0.0
    if "cuda" in device:
        peak_mem_mb = torch.cuda.max_memory_allocated(device=q.device) / (1024 ** 2)

    return {
        "name": name,
        "ms_per_iter": elapsed * 1000 / iters,
        "iters_per_sec": iters / elapsed,
        "peak_mem_mb": peak_mem_mb,
    }


def try_custom_impl() -> Optional[Callable[..., torch.Tensor]]:
    try:
        # 约定后续你的自研实现入口；当前不存在时会自动跳过
        from ops.flash_attention2 import flash_attention2  # type: ignore
        return flash_attention2
    except Exception:
        return None


@torch.no_grad()
def compare_numerical(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool,
    custom_fn: Optional[Callable[..., torch.Tensor]],
):
    ref = attention_sdpa(q, k, v, is_causal=is_causal, dropout_p=0.0, training=False)
    naive = attention_naive(q, k, v, is_causal=is_causal, dropout_p=0.0, training=False)
    err_naive = (naive - ref).abs().max().item()
    print(f"[Correctness] naive vs sdpa max_abs_err: {err_naive:.6e}")
    if custom_fn is not None:
        out = custom_fn(q, k, v, is_causal=is_causal, dropout_p=0.0, training=False)
        err_custom = (out - ref).abs().max().item()
        print(f"[Correctness] custom vs sdpa max_abs_err: {err_custom:.6e}")


def main():
    parser = argparse.ArgumentParser(description="Attention benchmark for Stage-1 FlashAttention study")
    parser.add_argument("--bsz", type=int, default=8)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--head_dim", type=int, default=64)
    parser.add_argument("--dtype", type=str, default="bf16", help="fp16|bf16|fp32")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--causal", action="store_true", default=True)
    parser.add_argument("--dropout_p", type=float, default=0.0)
    parser.add_argument("--backward", action="store_true", help="同时测试前反向")
    parser.add_argument("--include_custom", action="store_true", help="包含自研实现（若已实现）")
    args = parser.parse_args()

    dtype = parse_dtype(args.dtype)
    requires_grad = bool(args.backward)
    q, k, v = make_inputs(
        bsz=args.bsz,
        n_heads=args.heads,
        seq_len=args.seq_len,
        head_dim=args.head_dim,
        dtype=dtype,
        device=args.device,
        requires_grad=requires_grad,
    )

    print(
        f"[Config] bsz={args.bsz}, heads={args.heads}, seq_len={args.seq_len}, head_dim={args.head_dim}, "
        f"dtype={dtype}, backward={args.backward}, device={args.device}"
    )

    results = []
    results.append(
        benchmark_impl(
            name="naive",
            fn=attention_naive,
            q=q,
            k=k,
            v=v,
            warmup=args.warmup,
            iters=args.iters,
            is_causal=args.causal,
            dropout_p=args.dropout_p,
            backward=args.backward,
        )
    )
    results.append(
        benchmark_impl(
            name="torch_sdpa",
            fn=attention_sdpa,
            q=q,
            k=k,
            v=v,
            warmup=args.warmup,
            iters=args.iters,
            is_causal=args.causal,
            dropout_p=args.dropout_p,
            backward=args.backward,
        )
    )

    custom_fn = try_custom_impl() if args.include_custom else None
    if custom_fn is not None:
        results.append(
            benchmark_impl(
                name="custom_fa2",
                fn=custom_fn,
                q=q,
                k=k,
                v=v,
                warmup=args.warmup,
                iters=args.iters,
                is_causal=args.causal,
                dropout_p=args.dropout_p,
                backward=args.backward,
            )
        )
    elif args.include_custom:
        print("[Info] custom_fa2 未找到，已跳过。后续在 ops/flash_attention2.py 中实现 flash_attention2 即可自动接入。")

    print("\n=== Benchmark Results ===")
    for r in results:
        print(
            f"{r['name']:>12} | {r['ms_per_iter']:.3f} ms/iter | {r['iters_per_sec']:.2f} iter/s | "
            f"{r['peak_mem_mb']:.1f} MB peak"
        )

    # 数值对齐建议只在无 dropout 的前向下做
    if args.dropout_p == 0.0 and not args.backward:
        with torch.no_grad():
            compare_numerical(q, k, v, is_causal=args.causal, custom_fn=custom_fn)


if __name__ == "__main__":
    main()
