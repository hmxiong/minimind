import math

import torch
import torch.nn.functional as F


def flash_attention2_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = True,
    dropout_p: float = 0.0,
    training: bool = False,
    block_q: int = 128,
    block_k: int = 128,
) -> torch.Tensor:
    """
    FlashAttention-2 的 PyTorch 参考版（前向）:
    - 使用 online softmax 的块归并逻辑，避免显式构建完整 SxS 注意力矩阵。
    - 目标是便于学习和验证正确性，不追求性能。

    输入张量形状:
      q, k, v: [B, H, S, D]
    返回:
      out: [B, H, S, D]
    """
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q/k/v must be 4D tensors with shape [B, H, S, D]")
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError("q, k, v must have the same shape")
    if q.size(-1) <= 0:
        raise ValueError("head_dim must be > 0")

    bsz, n_heads, seq_len, head_dim = q.shape
    scale = 1.0 / math.sqrt(head_dim)

    # 用 fp32 做归并统计更稳定，最终再 cast 回输入 dtype
    out = torch.zeros_like(q, dtype=torch.float32)

    for q_start in range(0, seq_len, block_q):
        q_end = min(q_start + block_q, seq_len)
        q_blk = q[:, :, q_start:q_end, :]  # [B,H,Qb,D]
        q_blk_fp32 = q_blk.float()
        q_len = q_end - q_start

        # online softmax 状态: m_i, l_i, acc_i
        m_i = torch.full((bsz, n_heads, q_len), -float("inf"), device=q.device, dtype=torch.float32)
        l_i = torch.zeros((bsz, n_heads, q_len), device=q.device, dtype=torch.float32)
        acc_i = torch.zeros((bsz, n_heads, q_len, head_dim), device=q.device, dtype=torch.float32)

        for k_start in range(0, seq_len, block_k):
            k_end = min(k_start + block_k, seq_len)
            k_blk = k[:, :, k_start:k_end, :]  # [B,H,Kb,D]
            v_blk = v[:, :, k_start:k_end, :]  # [B,H,Kb,D]

            scores = torch.matmul(q_blk_fp32, k_blk.float().transpose(-2, -1)) * scale  # [B,H,Qb,Kb]

            if is_causal:
                # 全局位置索引决定因果掩码
                q_pos = torch.arange(q_start, q_end, device=q.device)[:, None]  # [Qb,1]
                k_pos = torch.arange(k_start, k_end, device=q.device)[None, :]  # [1,Kb]
                causal_mask = k_pos > q_pos  # [Qb,Kb]
                scores = scores.masked_fill(causal_mask[None, None, :, :], -float("inf"))

            # 块内统计
            m_blk = scores.max(dim=-1).values  # [B,H,Qb]
            p = torch.exp(scores - m_blk.unsqueeze(-1))  # [B,H,Qb,Kb]
            l_blk = p.sum(dim=-1)  # [B,H,Qb]

            if dropout_p > 0.0 and training:
                p = F.dropout(p, p=dropout_p, training=True)

            pv_blk = torch.matmul(p, v_blk.float())  # [B,H,Qb,D]

            # 块间 online 合并
            m_new = torch.maximum(m_i, m_blk)
            alpha = torch.exp(m_i - m_new)  # [B,H,Qb]
            beta = torch.exp(m_blk - m_new)  # [B,H,Qb]
            l_new = alpha * l_i + beta * l_blk

            acc_i = acc_i * alpha.unsqueeze(-1) + pv_blk * beta.unsqueeze(-1)
            m_i, l_i = m_new, l_new

        out[:, :, q_start:q_end, :] = acc_i / (l_i.unsqueeze(-1) + 1e-20)

    return out.to(q.dtype)


# 统一导出名，方便 benchmark 的自定义实现自动接入
def flash_attention2(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = True,
    dropout_p: float = 0.0,
    training: bool = False,
):
    return flash_attention2_ref(
        q=q,
        k=k,
        v=v,
        is_causal=is_causal,
        dropout_p=dropout_p,
        training=training,
    )
