import math

import pytest
import torch
from torch.nn import functional as F

from train.losses import relation_kl, split_relation_heads

RELATIONS = {(1, 1): 1.0, (2, 2): 1.0, (3, 3): 1.0}


def reference_kl_relation(rel_T, rel_S, mask):
    """The original `_kl_relation` from minilmv2_fast.py, kept verbatim as the
    golden reference for the rewritten, chunked implementation."""
    B, A_r, S, _ = rel_T.shape
    mask_b = mask.to(torch.bool)
    q_mask = mask_b[:, None, :, None]
    k_mask = mask_b[:, None, None, :]
    m = q_mask & k_mask
    neg_inf = torch.finfo(rel_T.dtype).min
    rel_T_m = rel_T.masked_fill(~m, neg_inf)
    rel_S_m = rel_S.masked_fill(~m, neg_inf)
    p = F.softmax(rel_T_m, dim=-1)
    q_log = F.log_softmax(rel_S_m, dim=-1)
    kl_el = F.kl_div(q_log, p, reduction="none")
    kl_sum = (kl_el * m).sum(dim=(-1, -2))
    L = mask_b.sum(dim=-1).clamp(min=1)
    loss_b = kl_sum.sum(dim=-1) / (A_r * L)
    return loss_b.mean()


def make_qkv(seed, B=3, A_r=4, S=8, d_r=8, requires_grad=False):
    g = torch.Generator().manual_seed(seed)
    return tuple(
        torch.randn(B, A_r, S, d_r, generator=g, requires_grad=requires_grad)
        for _ in range(3)
    )


def ragged_mask(B=3, S=8, lengths=(8, 5, 3)):
    mask = torch.zeros(B, S, dtype=torch.long)
    for row, length in enumerate(lengths):
        mask[row, :length] = 1
    return mask


def reference_loss(qkv_T, qkv_S, mask, relations=RELATIONS):
    total = torch.zeros(())
    sqrt_T = math.sqrt(qkv_T[0].size(-1))
    sqrt_S = math.sqrt(qkv_S[0].size(-1))
    for (i, j), w in relations.items():
        rel_T = qkv_T[i - 1] @ qkv_T[j - 1].transpose(-1, -2) / sqrt_T
        rel_S = qkv_S[i - 1] @ qkv_S[j - 1].transpose(-1, -2) / sqrt_S
        total = total + w * reference_kl_relation(rel_T.detach(), rel_S, mask)
    return total


def test_matches_original_implementation():
    qkv_T, qkv_S = make_qkv(0), make_qkv(1)
    mask = ragged_mask()
    expected = reference_loss(qkv_T, qkv_S, mask)
    actual = relation_kl(qkv_T, qkv_S, RELATIONS, mask, head_chunk_size=None)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("chunk", [1, 2, 3, 8])
def test_chunked_equals_unchunked(chunk):
    qkv_T, qkv_S = make_qkv(2), make_qkv(3)
    mask = ragged_mask()
    full = relation_kl(qkv_T, qkv_S, RELATIONS, mask, head_chunk_size=None)
    chunked = relation_kl(qkv_T, qkv_S, RELATIONS, mask, head_chunk_size=chunk)
    torch.testing.assert_close(chunked, full, rtol=1e-6, atol=1e-7)


def test_pad_invariance():
    """Randomizing Q/K/V content at padded positions must not change the loss."""
    qkv_T, qkv_S = make_qkv(4), make_qkv(5)
    mask = ragged_mask()
    base = relation_kl(qkv_T, qkv_S, RELATIONS, mask)

    pad = (~mask.bool())[:, None, :, None]  # (B, 1, S, 1)
    scrambled_T = tuple(torch.where(pad, torch.randn_like(t) * 10, t) for t in qkv_T)
    scrambled_S = tuple(torch.where(pad, torch.randn_like(t) * 10, t) for t in qkv_S)
    scrambled = relation_kl(scrambled_T, scrambled_S, RELATIONS, mask)
    torch.testing.assert_close(scrambled, base, rtol=1e-6, atol=1e-7)


def test_zero_for_identical_relations():
    qkv = make_qkv(6)
    mask = ragged_mask()
    loss = relation_kl(qkv, qkv, RELATIONS, mask)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_gradients_flow_to_student_only():
    qkv_T = make_qkv(7)
    qkv_S = make_qkv(8, requires_grad=True)
    mask = ragged_mask()
    loss = relation_kl(qkv_T, qkv_S, RELATIONS, mask)
    loss.backward()
    for t in qkv_S:
        assert t.grad is not None and torch.isfinite(t.grad).all()
        # padded positions receive no gradient
        assert (t.grad * (~mask.bool())[:, None, :, None]).abs().sum() == 0


def test_relation_weights_scale_loss():
    qkv_T, qkv_S = make_qkv(9), make_qkv(10)
    mask = ragged_mask()
    single = relation_kl(qkv_T, qkv_S, {(1, 1): 1.0}, mask)
    double = relation_kl(qkv_T, qkv_S, {(1, 1): 2.0}, mask)
    torch.testing.assert_close(double, single * 2)


def test_bf16_inputs_accepted():
    qkv_T = tuple(t.bfloat16() for t in make_qkv(11))
    qkv_S = tuple(t.bfloat16() for t in make_qkv(12))
    mask = ragged_mask()
    loss = relation_kl(qkv_T, qkv_S, RELATIONS, mask)
    assert loss.dtype == torch.float32 and torch.isfinite(loss)


def test_autocast_does_not_change_loss():
    """The loss must stay fp32 under autocast (e.g. the Trainer's --bf16):
    autocast would otherwise silently downcast the relation matmuls."""
    qkv_T, qkv_S = make_qkv(13), make_qkv(14)
    mask = ragged_mask()
    plain = relation_kl(qkv_T, qkv_S, RELATIONS, mask)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        autocast = relation_kl(qkv_T, qkv_S, RELATIONS, mask)
    torch.testing.assert_close(autocast, plain, rtol=0, atol=0)


def test_cross_relations_supported():
    """Q-K / Q-V / K-V relations (the v1-style set) must work too."""
    qkv_T, qkv_S = make_qkv(15), make_qkv(16)
    mask = ragged_mask()
    loss = relation_kl(qkv_T, qkv_S, {(1, 2): 1.0, (1, 3): 1.0, (2, 3): 1.0}, mask)
    assert torch.isfinite(loss) and loss.item() > 0


def test_chunk_larger_than_head_count():
    qkv_T, qkv_S = make_qkv(17), make_qkv(18)
    mask = ragged_mask()
    full = relation_kl(qkv_T, qkv_S, RELATIONS, mask, head_chunk_size=None)
    oversized = relation_kl(qkv_T, qkv_S, RELATIONS, mask, head_chunk_size=999)
    torch.testing.assert_close(oversized, full)


def test_no_padding_batch():
    qkv_T, qkv_S = make_qkv(19), make_qkv(20)
    mask = torch.ones(3, 8, dtype=torch.long)
    loss = relation_kl(qkv_T, qkv_S, RELATIONS, mask)
    assert torch.isfinite(loss) and loss.item() > 0


def test_batch_size_one():
    qkv_T, qkv_S = make_qkv(21, B=1), make_qkv(22, B=1)
    mask = ragged_mask(B=1, lengths=(5,))
    loss = relation_kl(qkv_T, qkv_S, RELATIONS, mask)
    assert torch.isfinite(loss)


def test_mismatched_teacher_student_shapes_raise():
    qkv_T = make_qkv(23, A_r=4)
    qkv_S = make_qkv(24, A_r=2)
    with pytest.raises(ValueError, match="disagree"):
        relation_kl(qkv_T, qkv_S, RELATIONS, ragged_mask())


def test_different_head_dims_allowed():
    """Teacher and student may have different d_r - only (B, A_r, S) must match."""
    qkv_T = make_qkv(25, d_r=16)
    qkv_S = make_qkv(26, d_r=4)
    loss = relation_kl(qkv_T, qkv_S, RELATIONS, ragged_mask())
    assert torch.isfinite(loss) and loss.item() > 0


def test_split_relation_heads_shape_and_validation():
    x = torch.randn(2, 5, 32)
    out = split_relation_heads(x, 4)
    assert out.shape == (2, 4, 5, 8)
    torch.testing.assert_close(out[0, 1, 2], x[0, 2, 8:16])
    with pytest.raises(ValueError):
        split_relation_heads(x, 5)
