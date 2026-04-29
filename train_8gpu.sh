cd /Users/xhm/Desktop/ProjectWorkSpace/MLsys/minimind/trainer

torchrun --standalone --nproc_per_node=8 train_pretrain_fsdp.py \
  --data_path ../dataset/pretrain_t2t_mini.jsonl \
  --dtype bfloat16 \
  --sharding full \
  --hidden_size 1344 \
  --num_hidden_layers 16 \
  --max_seq_len 1024 \
  --batch_size 12 \
  --accumulation_steps 2 \
  --num_workers 12 \
  --save_weight pretrain_fsdp
