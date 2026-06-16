"""Distillation losses.

The composable terms a distiller can mix (each masked to real tokens):
- `relation_kl`     - MiniLMv2 self-attention relation distillation (the core).
- `hidden_state_mse` - TinyBERT-style representation matching (helps tasks that
  read a pooled representation, e.g. text classification).
- `logit_kd`        - temperature-scaled KL over output (MLM) logits.
"""
from __future__ import annotations

import contextlib
import math
from typing import Dict, Sequence, Tuple

import torch
from torch.nn import functional as F

Relations = Dict[Tuple[int, int], float]


def _autocast_off(device: torch.device):
    """Disable autocast for the loss computation.

    The loss is deliberately computed in fp32; without this, running under the
    Trainer's bf16 autocast would silently downcast the relation matmuls and
    softmaxes back to bf16.
    """
    if device.type in ("cuda", "cpu"):
        return torch.autocast(device_type=device.type, enabled=False)
    return contextlib.nullcontext()


def split_relation_heads(x: torch.Tensor, num_relation_heads: int) -> torch.Tensor:
    """Reshape (B, S, H) projections into (B, A_r, S, d_r) relation heads."""
    if x.dim() != 3:
        raise ValueError(f"Expected a (B, S, H) tensor, got shape {tuple(x.shape)}")
    B, S, H = x.shape
    if H % num_relation_heads != 0:
        raise ValueError(
            f"Hidden size {H} is not divisible by num_relation_heads {num_relation_heads}"
        )
    d_r = H // num_relation_heads
    return x.view(B, S, num_relation_heads, d_r).permute(0, 2, 1, 3)


def relation_kl(
    qkv_teacher: Sequence[torch.Tensor],
    qkv_student: Sequence[torch.Tensor],
    relations: Relations,
    attention_mask: torch.Tensor,
    head_chunk_size: int | None = 8,
    segment_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Masked KL between teacher and student self-attention relations (MiniLMv2 eq. 6).

    Arguments
    ---------
    qkv_teacher, qkv_student : (q, k, v) tensors of shape (B, A_r, S, d_r).
    relations               : {(i, j): weight} with 1=Q, 2=K, 3=V.
    attention_mask          : (B, S), 1 = real token, 0 = padding.
    head_chunk_size         : relation heads processed per chunk. The (B, A_r, S, S)
                              relation matrices dominate peak memory, so they are
                              materialized only `head_chunk_size` heads at a time.
                              None processes all heads at once.
    segment_ids             : (B, S) optional; for packed rows, relations are
                              restricted to within-segment pairs and each row
                              is normalized by its total real-token count.

    Returns a scalar fp32 loss. Inputs may be bf16/fp16; each chunk is computed
    in fp32 for numerical parity with the original fp32 implementation.
    """
    q_t = qkv_teacher[0]
    B, A_r, S, d_teacher = q_t.shape
    d_student = qkv_student[0].shape[-1]
    if qkv_student[0].shape[:3] != (B, A_r, S):
        raise ValueError(
            f"Teacher/student relation-head shapes disagree: "
            f"{tuple(q_t.shape[:3])} vs {tuple(qkv_student[0].shape[:3])}"
        )

    mask = attention_mask.to(device=q_t.device, dtype=torch.bool)
    if segment_ids is None:
        key_mask = mask[:, None, None, :]    # (B, 1, 1, S)
    else:
        same_segment = segment_ids[:, :, None] == segment_ids[:, None, :]
        key_mask = (same_segment & mask[:, :, None] & mask[:, None, :])[:, None]  # (B, 1, S, S)
    query_mask = mask[:, None, :, None]  # (B, 1, S, 1)
    lengths = mask.sum(dim=-1).clamp(min=1)  # (B,)

    chunk = A_r if head_chunk_size is None else max(1, head_chunk_size)
    neg_inf = torch.finfo(torch.float32).min

    loss = q_t.new_zeros((), dtype=torch.float32)
    with _autocast_off(q_t.device):
        for (i, j), weight in relations.items():
            T1, T2 = qkv_teacher[i - 1], qkv_teacher[j - 1]
            S1, S2 = qkv_student[i - 1], qkv_student[j - 1]

            kl_sum = q_t.new_zeros(B, dtype=torch.float32)
            for h0 in range(0, A_r, chunk):
                h1 = min(h0 + chunk, A_r)
                logits_t = (
                    T1[:, h0:h1].float() @ T2[:, h0:h1].float().transpose(-1, -2)
                ) / math.sqrt(d_teacher)
                logits_s = (
                    S1[:, h0:h1].float() @ S2[:, h0:h1].float().transpose(-1, -2)
                ) / math.sqrt(d_student)

                logits_t = logits_t.masked_fill(~key_mask, neg_inf)
                logits_s = logits_s.masked_fill(~key_mask, neg_inf)

                logp_t = F.log_softmax(logits_t.detach(), dim=-1)
                logp_s = F.log_softmax(logits_s, dim=-1)
                p_t = logp_t.exp()

                # p_t is exactly 0 at masked keys; guard the 0 * (-inf - -inf) = nan case.
                kl = torch.where(p_t > 0, p_t * (logp_t - logp_s), p_t.new_zeros(()))
                kl = kl * query_mask
                kl_sum = kl_sum + kl.sum(dim=(-3, -2, -1))

            loss_per_example = kl_sum / (A_r * lengths)
            loss = loss + weight * loss_per_example.mean()
    return loss


def hidden_state_mse(
    hidden_teacher: torch.Tensor,
    hidden_student: torch.Tensor,
    attention_mask: torch.Tensor,
    projection: torch.nn.Module | None = None,
) -> torch.Tensor:
    """Masked MSE between teacher and student hidden states (TinyBERT-style).

    `hidden_*` are (B, S, H) layer outputs; the student is mapped to the
    teacher's width by `projection` (a Linear, the only trainable part of this
    term - discarded at export). MSE is over real tokens only. Targets are
    detached; computed in fp32. Helps tasks that read a pooled representation
    (e.g. text classification), which relations alone do not directly transfer.
    """
    student = hidden_student if projection is None else projection(hidden_student)
    mask = attention_mask.to(device=student.device, dtype=torch.bool)[..., None]  # (B, S, 1)
    with _autocast_off(student.device):
        diff = (student.float() - hidden_teacher.float().detach()) ** 2
        diff = diff * mask
        denom = mask.sum().clamp(min=1) * student.shape[-1]
        return diff.sum() / denom


def logit_kd(
    logits_teacher: torch.Tensor,
    logits_student: torch.Tensor,
    attention_mask: torch.Tensor,
    temperature: float = 2.0,
    seq_chunk_size: int | None = None,
) -> torch.Tensor:
    """Temperature-scaled KL over output (MLM) logits at real tokens.

    `logits_*` are (B, S, V) over a SHARED vocabulary. Standard KD: KL of the
    softened teacher distribution into the student, scaled by T^2. Teacher is
    detached; computed in fp32.

    `seq_chunk_size` bounds peak memory: the fp32 softmax/KL is materialized at
    most `(B, chunk, V)` at a time rather than `(B, S, V)`. For a large vocab
    (e.g. 256k) this is what lets the batch size grow. None = whole sequence.
    """
    mask = attention_mask.to(device=logits_student.device, dtype=torch.bool)
    B, S, _ = logits_student.shape
    chunk = S if seq_chunk_size is None else max(1, seq_chunk_size)
    t = temperature
    total = logits_student.new_zeros((), dtype=torch.float32)
    with _autocast_off(logits_student.device):
        for s0 in range(0, S, chunk):
            s1 = min(s0 + chunk, S)
            logp_s = F.log_softmax(logits_student[:, s0:s1].float() / t, dim=-1)
            logp_t = F.log_softmax(logits_teacher[:, s0:s1].float().detach() / t, dim=-1)
            p_t = logp_t.exp()
            kl = (p_t * (logp_t - logp_s)).sum(dim=-1)  # (B, chunk)
            total = total + (kl * mask[:, s0:s1]).sum()
        return (t * t) * total / mask.sum().clamp(min=1)
