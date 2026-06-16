from collections import Counter

import pytest
import torch

from train.distiller import MiniLMDistiller
from train.vocab_pruning import apply_vocab_pruning, count_token_ids, select_vocab

from conftest import PAD_ID, tiny_bert

RELATIONS = {(1, 1): 1.0, (2, 2): 1.0, (3, 3): 1.0}


def test_count_token_ids():
    examples = [{"input_ids": [1, 2, 2]}, {"input_ids": torch.tensor([2, 3])}, {"input_ids": [9]}]
    counts = count_token_ids(iter(examples), num_docs=2)
    assert counts == Counter({2: 3, 1: 1, 3: 1})


def test_select_vocab_top_k_plus_specials():
    counts = Counter({10: 100, 11: 50, 12: 10, 13: 1})
    keep = select_vocab(counts, keep_size=3, must_keep=[0, 5])
    assert keep == [0, 5, 10]  # specials always kept, then most frequent


def test_select_vocab_by_coverage():
    from train.vocab_pruning import select_vocab_by_coverage

    counts = Counter({10: 70, 11: 20, 12: 9, 13: 1})
    # 70% needs just id 10; 90% needs 10+11; must_keep always included
    assert select_vocab_by_coverage(counts, 0.70, [0]) == [0, 10]
    assert select_vocab_by_coverage(counts, 0.90, [0]) == [0, 10, 11]
    assert select_vocab_by_coverage(counts, 1.0, [0]) == [0, 10, 11, 12, 13]
    with pytest.raises(ValueError):
        select_vocab_by_coverage(counts, 1.5, [0])


def test_coverage_counts_must_keep_toward_coverage():
    from train.vocab_pruning import select_vocab_by_coverage

    counts = Counter({10: 50, 11: 50})
    # must_keep id 10 already covers 50%; threshold 0.5 adds nothing
    assert select_vocab_by_coverage(counts, 0.5, [10]) == [10]


def test_apply_vocab_pruning_remap_and_rows(student):
    old_weight = student.embeddings.word_embeddings.weight.detach().clone()
    keep = [PAD_ID, 1, 7, 42]
    remap = apply_vocab_pruning(student, keep, unk_id=1)

    assert student.config.vocab_size == 4
    assert student.embeddings.word_embeddings.num_embeddings == 4
    torch.testing.assert_close(
        student.embeddings.word_embeddings.weight.detach(), old_weight[torch.tensor(keep)]
    )
    assert remap[42] == 3 and remap[7] == 2 and remap[PAD_ID] == 0
    assert remap[55] == 1  # unseen id -> unk's new position
    assert student.config.pad_token_id == 0


def test_apply_vocab_pruning_validates_specials(student):
    with pytest.raises(ValueError, match="unk_id"):
        apply_vocab_pruning(student, [PAD_ID, 5], unk_id=99)
    with pytest.raises(ValueError, match="pad_token_id"):
        apply_vocab_pruning(student, [5, 99], unk_id=99)


def test_pruned_student_loss_identical_when_coverage_full(teacher, student, batch):
    """Pruning to a superset of the batch's ids (with copied rows) must not
    change the loss at all - the student computes on identical embeddings."""
    torch.manual_seed(0)
    distiller_args = dict(L=2, M=2, relations=RELATIONS, A_r=4, teacher_dtype=torch.float32)
    (base,) = MiniLMDistiller(teacher=teacher, student=student, **distiller_args)(**batch)

    used_ids = sorted(set(batch["input_ids"].flatten().tolist()) | {PAD_ID, 1})
    remap = apply_vocab_pruning(student, used_ids, unk_id=1)
    pruned = MiniLMDistiller(
        teacher=teacher, student=student, student_input_remap=remap, **distiller_args
    )
    (loss,) = pruned(**batch)
    torch.testing.assert_close(loss, base)


def test_pruning_composes_with_packing(teacher, student):
    """Pruned student + packed row: the remap must apply to the student while
    the teacher sees original ids, and the result must equal the per-document
    combination - both features active at once."""
    torch.manual_seed(0)
    l1, l2, S = 6, 4, 12
    doc1 = torch.randint(2, 99, (1, l1))
    doc2 = torch.randint(2, 99, (1, l2))
    used = sorted(set(torch.cat([doc1, doc2], 1).flatten().tolist()) | {PAD_ID, 1})
    remap = apply_vocab_pruning(student, used, unk_id=1)

    args = dict(L=2, M=2, relations=RELATIONS, A_r=4, teacher_dtype=torch.float32,
                student_input_remap=remap)
    distiller = MiniLMDistiller(teacher=teacher, student=student, **args)

    packed_ids = torch.cat([doc1, doc2, torch.zeros(1, S - l1 - l2, dtype=torch.long)], dim=1)
    mask = torch.tensor([[1] * (l1 + l2) + [0] * (S - l1 - l2)])
    seg = torch.tensor([[0] * l1 + [1] * l2 + [-1] * (S - l1 - l2)])
    pos = torch.tensor([[*range(l1), *range(l2)] + [0] * (S - l1 - l2)])

    (packed,) = distiller(input_ids=packed_ids, attention_mask=mask, segment_ids=seg, position_ids=pos)
    (loss1,) = distiller(input_ids=doc1, attention_mask=torch.ones(1, l1, dtype=torch.long))
    (loss2,) = distiller(input_ids=doc2, attention_mask=torch.ones(1, l2, dtype=torch.long))
    expected = (loss1 * 4 * l1 + loss2 * 4 * l2) / (4 * (l1 + l2))
    torch.testing.assert_close(packed, expected, rtol=1e-4, atol=1e-5)


def test_unseen_id_maps_to_unk(teacher, student, batch):
    used_ids = sorted(set(batch["input_ids"].flatten().tolist()) | {PAD_ID, 1})
    remap = apply_vocab_pruning(student, used_ids, unk_id=1)
    distiller = MiniLMDistiller(
        teacher=teacher, student=student, L=2, M=2, relations=RELATIONS,
        A_r=4, teacher_dtype=torch.float32, student_input_remap=remap,
    )
    ids = batch["input_ids"].clone()
    ids[0, 0] = 98  # id outside the pruned vocab
    (loss,) = distiller(input_ids=ids, attention_mask=batch["attention_mask"])
    assert torch.isfinite(loss)
