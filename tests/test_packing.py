import math

import pytest
import torch

from train.distiller import MiniLMDistiller
from train.losses import relation_kl
from train.packing import PackFn

from conftest import tiny_bert

RELATIONS = {(1, 1): 1.0, (2, 2): 1.0, (3, 3): 1.0}


def test_packfn_packs_and_pads():
    fn = PackFn(max_seq_len=10, pad_token_id=0)
    out = fn({"input_ids": [[1, 2, 3], [4, 5, 6, 7], [8, 9], [10, 11, 12, 13, 14]]})
    assert all(len(row) == 10 for col in out.values() for row in col)
    # docs 1-3 (3+4+2=9 tokens) fit row 0; doc 4 starts row 1
    assert out["input_ids"][0][:9] == [1, 2, 3, 4, 5, 6, 7, 8, 9]
    assert out["segment_ids"][0][:9] == [0, 0, 0, 1, 1, 1, 1, 2, 2]
    assert out["position_ids"][0][:9] == [0, 1, 2, 0, 1, 2, 3, 0, 1]
    assert out["attention_mask"][0] == [1] * 9 + [0]
    assert out["segment_ids"][0][9] == -1
    assert out["input_ids"][1][:5] == [10, 11, 12, 13, 14]


def test_packfn_preserves_all_tokens():
    docs = [[i] * (i % 7 + 1) for i in range(1, 40)]
    out = PackFn(max_seq_len=16, pad_token_id=0)({"input_ids": docs})
    packed_tokens = [
        t for row, m in zip(out["input_ids"], out["attention_mask"])
        for t, real in zip(row, m) if real
    ]
    assert packed_tokens == [t for d in docs for t in d]


def test_packfn_oversized_doc_truncated():
    out = PackFn(max_seq_len=4, pad_token_id=0)({"input_ids": [[1, 2, 3, 4, 5, 6]]})
    assert out["input_ids"] == [[1, 2, 3, 4]]


def make_qkv(seed, B, S, A_r=4, d_r=8):
    g = torch.Generator().manual_seed(seed)
    return tuple(torch.randn(B, A_r, S, d_r, generator=g) for _ in range(3))


def test_segmented_loss_equals_unpacked_loss():
    """A packed row of two documents must yield the same KL mass as the two
    documents scored separately."""
    l1, l2, S = 5, 3, 10
    A_r = 4
    qkv_T, qkv_S = make_qkv(0, 1, S), make_qkv(1, 1, S)
    mask = torch.tensor([[1] * (l1 + l2) + [0] * (S - l1 - l2)])
    seg = torch.tensor([[0] * l1 + [1] * l2 + [-1] * (S - l1 - l2)])

    packed = relation_kl(qkv_T, qkv_S, RELATIONS, mask, segment_ids=seg)

    def doc_loss(s0, s1, length):
        sl = lambda qkv: tuple(t[:, :, s0:s1] for t in qkv)
        m = torch.ones(1, length, dtype=torch.long)
        return relation_kl(sl(qkv_T), sl(qkv_S), RELATIONS, m)

    # per-doc losses are kl_i / (A_r * l_i); the packed row is sum(kl_i) / (A_r * (l1+l2))
    kl1 = doc_loss(0, l1, l1) * A_r * l1
    kl2 = doc_loss(l1, l1 + l2, l2) * A_r * l2
    expected = (kl1 + kl2) / (A_r * (l1 + l2))
    torch.testing.assert_close(packed, expected, rtol=1e-5, atol=1e-6)


def test_segmented_loss_cross_segment_invariance():
    """Changing one document's content must not change the other's KL mass."""
    l1, l2 = 4, 4
    S = l1 + l2
    A_r = 4
    qkv_T, qkv_S = make_qkv(2, 1, S), make_qkv(3, 1, S)
    mask = torch.ones(1, S, dtype=torch.long)
    seg = torch.tensor([[0] * l1 + [1] * l2])

    def kl_of_doc1(qkv_T, qkv_S):
        total = relation_kl(qkv_T, qkv_S, RELATIONS, mask, segment_ids=seg) * A_r * S
        sl = lambda qkv: tuple(t[:, :, l1:] for t in qkv)
        m2 = torch.ones(1, l2, dtype=torch.long)
        kl2 = relation_kl(sl(qkv_T), sl(qkv_S), RELATIONS, m2) * A_r * l2
        return total - kl2

    base = kl_of_doc1(qkv_T, qkv_S)
    scramble = lambda qkv: tuple(
        torch.cat([t[:, :, :l1], torch.randn_like(t[:, :, l1:])], dim=2) for t in qkv
    )
    torch.manual_seed(7)
    perturbed = kl_of_doc1(scramble(qkv_T), scramble(qkv_S))
    torch.testing.assert_close(perturbed, base, rtol=1e-5, atol=1e-6)


def test_distiller_packed_equals_separate(teacher, student):
    """End-to-end: a packed row through the distiller must equal the
    token-weighted combination of the documents run separately."""
    torch.manual_seed(0)
    distiller = MiniLMDistiller(
        teacher=teacher, student=student, L=2, M=2,
        relations=RELATIONS, A_r=4, teacher_dtype=torch.float32,
    )
    l1, l2, S = 6, 4, 12
    doc1 = torch.randint(1, 99, (1, l1))
    doc2 = torch.randint(1, 99, (1, l2))

    packed_ids = torch.cat([doc1, doc2, torch.zeros(1, S - l1 - l2, dtype=torch.long)], dim=1)
    mask = torch.tensor([[1] * (l1 + l2) + [0] * (S - l1 - l2)])
    seg = torch.tensor([[0] * l1 + [1] * l2 + [-1] * (S - l1 - l2)])
    pos = torch.tensor([list(range(l1)) + list(range(l2)) + [0] * (S - l1 - l2)])

    (packed,) = distiller(
        input_ids=packed_ids, attention_mask=mask, segment_ids=seg, position_ids=pos
    )
    (loss1,) = distiller(input_ids=doc1, attention_mask=torch.ones(1, l1, dtype=torch.long))
    (loss2,) = distiller(input_ids=doc2, attention_mask=torch.ones(1, l2, dtype=torch.long))

    expected = (loss1 * 4 * l1 + loss2 * 4 * l2) / (4 * (l1 + l2))
    torch.testing.assert_close(packed, expected, rtol=1e-4, atol=1e-5)


def test_distiller_packed_requires_position_ids(teacher, student):
    distiller = MiniLMDistiller(
        teacher=teacher, student=student, L=2, M=2,
        relations=RELATIONS, A_r=4, teacher_dtype=torch.float32,
    )
    ids = torch.randint(1, 99, (1, 8))
    with pytest.raises(ValueError, match="position_ids"):
        distiller(
            input_ids=ids,
            attention_mask=torch.ones(1, 8, dtype=torch.long),
            segment_ids=torch.zeros(1, 8, dtype=torch.long),
        )


def test_position_id_offset():
    from train.adapters import position_id_offset

    class Cfg:
        def __init__(self, mt, pad):
            self.model_type, self.pad_token_id = mt, pad

    assert position_id_offset(Cfg("bert", 0)) == 0
    assert position_id_offset(Cfg("xlm-roberta", 1)) == 2
    assert position_id_offset(Cfg("roberta", 1)) == 2
