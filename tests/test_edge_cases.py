"""Adversarial low-level edge-case tests for the MiniLMv2 distillation core.

Covers loss numerics, degenerate shapes/masks, dtype/layout robustness,
recorder lifecycle corner cases, and distiller state handling that the
mainline tests do not exercise. Packing/segment_ids is covered elsewhere.
"""
import math
import pickle

import pytest
import torch

from tests.conftest import PAD_ID, VOCAB, tiny_bert
from train.adapters import BertLikeRecorder
from train.distiller import MiniLMDistiller
from train.losses import relation_kl, split_relation_heads

RELATIONS = {(1, 1): 1.0, (2, 2): 1.0, (3, 3): 1.0}


def make_qkv(seed, B=2, A_r=2, S=6, d_r=4, dtype=torch.float32, requires_grad=False):
    g = torch.Generator().manual_seed(seed)
    return tuple(
        torch.randn(B, A_r, S, d_r, generator=g).to(dtype).requires_grad_(requires_grad)
        for _ in range(3)
    )


def make_distiller(teacher, student, **kwargs):
    defaults = dict(
        teacher=teacher, student=student, L=2, M=2,
        relations=RELATIONS, A_r=4, teacher_dtype=torch.float32,
    )
    defaults.update(kwargs)
    return MiniLMDistiller(**defaults)


# ---------------------------------------------------------------------------
# Loss numerics
# ---------------------------------------------------------------------------

def test_extreme_logit_magnitudes_stay_finite():
    """Q/K/V scaled by 100 produce relation logits ~1e4; softmax must not
    overflow and gradients must stay finite."""
    qkv_T = tuple(t * 100 for t in make_qkv(0))
    qkv_S = tuple((t * 100).detach().requires_grad_(True) for t in make_qkv(1))
    mask = torch.ones(2, 6, dtype=torch.long)
    loss = relation_kl(qkv_T, qkv_S, RELATIONS, mask)
    assert torch.isfinite(loss)
    loss.backward()
    for t in qkv_S:
        assert torch.isfinite(t.grad).all()


def test_extreme_logits_finite():
    """relation_kl must stay finite + differentiable under huge magnitudes."""
    qkv_T = tuple(torch.randn(2, 2, 6, 8, generator=torch.Generator().manual_seed(s)) * 1e3
                  for s in (1, 2, 3))
    qkv_S = tuple((torch.randn(2, 2, 6, 8, generator=torch.Generator().manual_seed(s)) * 1e3).requires_grad_(True)
                  for s in (4, 5, 6))
    mask = torch.ones(2, 6, dtype=torch.long)
    mask[1, 4:] = 0
    loss = relation_kl(qkv_T, qkv_S, {(1, 1): 1.0, (2, 2): 1.0, (3, 3): 1.0}, mask)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(torch.isfinite(t.grad).all() for t in qkv_S)


def test_length_one_sequence_loss_is_zero():
    """With a single real key, both distributions are point masses -> KL = 0."""
    qkv_T, qkv_S = make_qkv(2, S=1), make_qkv(3, S=1)
    mask = torch.ones(2, 1, dtype=torch.long)
    loss = relation_kl(qkv_T, qkv_S, RELATIONS, mask)
    assert torch.isfinite(loss)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_all_pad_row_contributes_exactly_zero():
    """Appending an all-pad row must not produce NaNs, and the loss must scale
    by exactly B/(B+1) (the pad row contributes 0 to the batch mean)."""
    qkv_T, qkv_S = make_qkv(4, B=2), make_qkv(5, B=2)
    mask = torch.ones(2, 6, dtype=torch.long)
    base = relation_kl(qkv_T, qkv_S, RELATIONS, mask)

    pad = lambda qkv: tuple(torch.cat([t, torch.randn_like(t[:1]) * 5]) for t in qkv)
    mask3 = torch.cat([mask, torch.zeros(1, 6, dtype=torch.long)])
    with_pad_row = relation_kl(pad(qkv_T), pad(qkv_S), RELATIONS, mask3)
    assert torch.isfinite(with_pad_row)
    torch.testing.assert_close(with_pad_row, base * 2 / 3, rtol=1e-6, atol=1e-7)


def test_all_pad_batch_is_zero_not_nan():
    qkv_T, qkv_S = make_qkv(6), make_qkv(7)
    mask = torch.zeros(2, 6, dtype=torch.long)
    loss = relation_kl(qkv_T, qkv_S, RELATIONS, mask)
    assert torch.isfinite(loss) and loss.item() == 0.0


def test_non_contiguous_qkv_inputs():
    """Tensors arriving via permute (as split_relation_heads produces) are
    non-contiguous; the loss must accept them unchanged."""
    qkv_T, qkv_S = make_qkv(8), make_qkv(9)
    mask = torch.ones(2, 6, dtype=torch.long)
    base = relation_kl(qkv_T, qkv_S, RELATIONS, mask)

    # materialize in (B, S, A_r, d_r) layout, view back via permute -> non-contiguous
    nc = lambda qkv: tuple(t.transpose(1, 2).contiguous().permute(0, 2, 1, 3) for t in qkv)
    assert not nc(qkv_T)[0].is_contiguous()
    loss = relation_kl(nc(qkv_T), nc(qkv_S), RELATIONS, mask)
    torch.testing.assert_close(loss, base, rtol=0, atol=0)


def test_fp16_inputs_accepted():
    qkv_T = tuple(t.half() for t in make_qkv(10))
    qkv_S = tuple(t.half() for t in make_qkv(11))
    mask = torch.ones(2, 6, dtype=torch.long)
    loss = relation_kl(qkv_T, qkv_S, RELATIONS, mask)
    assert loss.dtype == torch.float32 and torch.isfinite(loss)


def test_empty_relations_dict_gives_zero():
    qkv_T, qkv_S = make_qkv(12), make_qkv(13)
    mask = torch.ones(2, 6, dtype=torch.long)
    loss = relation_kl(qkv_T, qkv_S, {}, mask)
    assert loss.item() == 0.0 and loss.dtype == torch.float32


def test_zero_relation_weight_gives_zero():
    qkv_T, qkv_S = make_qkv(14), make_qkv(15)
    mask = torch.ones(2, 6, dtype=torch.long)
    loss = relation_kl(qkv_T, qkv_S, {(1, 1): 0.0}, mask)
    assert loss.item() == 0.0


def test_float_attention_mask_accepted():
    """HF collators sometimes emit float masks; result must match a long mask."""
    qkv_T, qkv_S = make_qkv(16), make_qkv(17)
    mask = torch.tensor([[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1]], dtype=torch.long)
    base = relation_kl(qkv_T, qkv_S, RELATIONS, mask)
    via_float = relation_kl(qkv_T, qkv_S, RELATIONS, mask.float())
    torch.testing.assert_close(via_float, base, rtol=0, atol=0)


def test_gradcheck_tiny_case():
    """Finite-difference gradient check. The loss internally computes in fp32
    (by design), so float64 inputs get loose tolerances."""
    torch.manual_seed(0)
    qkv_T = tuple(torch.randn(1, 1, 3, 2, dtype=torch.float64) for _ in range(3))
    qkv_S = tuple(
        torch.randn(1, 1, 3, 2, dtype=torch.float64, requires_grad=True)
        for _ in range(3)
    )
    mask = torch.tensor([[1, 1, 0]])

    def fn(*qkv_s):
        return relation_kl(qkv_T, qkv_s, RELATIONS, mask, head_chunk_size=None)

    assert torch.autograd.gradcheck(fn, qkv_S, eps=1e-3, atol=1e-3, rtol=5e-2)


def test_split_relation_heads_extremes():
    x = torch.randn(2, 5, 8)
    # A_r == hidden: each head has d_r == 1
    out = split_relation_heads(x, 8)
    assert out.shape == (2, 8, 5, 1)
    torch.testing.assert_close(out[:, 3, :, 0], x[:, :, 3])
    # A_r == 1: identity reshape with a head axis
    out1 = split_relation_heads(x, 1)
    assert out1.shape == (2, 1, 5, 8)
    torch.testing.assert_close(out1[:, 0], x)
    # wrong rank
    with pytest.raises(ValueError, match="B, S, H"):
        split_relation_heads(torch.randn(2, 5), 1)


# ---------------------------------------------------------------------------
# Recorders
# ---------------------------------------------------------------------------

def test_recorder_on_first_and_last_layer(teacher, batch):
    for layer in (1, 3):
        recorder = BertLikeRecorder(teacher, layer)
        with torch.no_grad():
            outputs = teacher(**batch, output_hidden_states=True)
        q, k, v = recorder.pop()
        attn = teacher.encoder.layer[layer - 1].attention.self
        layer_input = outputs.hidden_states[layer - 1]
        with torch.no_grad():
            torch.testing.assert_close(q, attn.query(layer_input))
            torch.testing.assert_close(k, attn.key(layer_input))
            torch.testing.assert_close(v, attn.value(layer_input))
        recorder.close()


def test_two_recorders_same_model(teacher, batch):
    """Two recorders on different layers of the same model must capture
    independent caches from a single forward pass."""
    r1 = BertLikeRecorder(teacher, 1)
    r2 = BertLikeRecorder(teacher, 3)
    with torch.no_grad():
        outputs = teacher(**batch, output_hidden_states=True)
    q1, _, _ = r1.pop()
    q3, _, _ = r2.pop()
    with torch.no_grad():
        attn1 = teacher.encoder.layer[0].attention.self
        attn3 = teacher.encoder.layer[2].attention.self
        torch.testing.assert_close(q1, attn1.query(outputs.hidden_states[0]))
        torch.testing.assert_close(q3, attn3.query(outputs.hidden_states[2]))
    assert not torch.equal(q1, q3)
    r1.close()
    r2.close()


def test_pop_after_forward_failing_before_hooked_layer(teacher, batch):
    """If the forward dies before the hooked layer runs, pop() must raise the
    'incomplete cache' error instead of returning stale/garbage tensors."""
    recorder = BertLikeRecorder(teacher, 3)

    def boom(*args, **kwargs):
        raise RuntimeError("injected failure")

    original = teacher.encoder.layer[1].forward
    teacher.encoder.layer[1].forward = boom
    try:
        with pytest.raises(RuntimeError, match="injected failure"):
            with torch.no_grad():
                teacher(**batch)
        with pytest.raises(RuntimeError, match="incomplete"):
            recorder.pop()
    finally:
        teacher.encoder.layer[1].forward = original
        recorder.close()


def test_failed_forward_does_not_leak_stale_qkv(teacher, batch):
    """A forward that fails *after* the hooked layer leaves a populated cache;
    the next successful forward must overwrite it so pop() returns fresh data."""
    recorder = BertLikeRecorder(teacher, 1)

    def boom(*args, **kwargs):
        raise RuntimeError("injected failure")

    original = teacher.encoder.layer[2].forward
    teacher.encoder.layer[2].forward = boom
    try:
        with pytest.raises(RuntimeError, match="injected failure"):
            with torch.no_grad():
                teacher(**batch)
    finally:
        teacher.encoder.layer[2].forward = original

    other_ids = batch["input_ids"].flip(0)
    other_mask = batch["attention_mask"].flip(0)
    with torch.no_grad():
        teacher(input_ids=other_ids, attention_mask=other_mask)
        q, _, _ = recorder.pop()
        attn = teacher.encoder.layer[0].attention.self
        expected_q = attn.query(
            teacher(input_ids=other_ids, attention_mask=other_mask,
                    output_hidden_states=True).hidden_states[0]
        )
        recorder._cache.clear()
    torch.testing.assert_close(q, expected_q)
    recorder.close()


def test_truncate_layers_idempotent(teacher, batch):
    recorder = BertLikeRecorder(teacher, 2)
    recorder.truncate_layers()
    assert len(teacher.encoder.layer) == 2
    recorder.truncate_layers()  # second call must be a no-op
    assert len(teacher.encoder.layer) == 2
    assert teacher.config.num_hidden_layers == 2
    with torch.no_grad():
        teacher(**batch)
    q, k, v = recorder.pop()
    assert q.shape[-1] == teacher.config.hidden_size
    recorder.close()


def test_non_contiguous_input_ids_through_distiller(teacher, student):
    torch.manual_seed(3)
    ids_t = torch.randint(1, VOCAB, (6, 2))  # (S, B)
    input_ids = ids_t.t()  # non-contiguous (B, S)
    assert not input_ids.is_contiguous()
    mask = torch.ones(2, 6, dtype=torch.long).t().t()
    distiller = make_distiller(teacher, student)
    (loss,) = distiller(input_ids=input_ids, attention_mask=mask)
    assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# Distiller
# ---------------------------------------------------------------------------

def test_distiller_L1_M1(batch):
    torch.manual_seed(0)
    teacher = tiny_bert(hidden_size=32, num_layers=1, num_heads=4).eval()
    torch.manual_seed(1)
    student = tiny_bert(hidden_size=16, num_layers=1, num_heads=2).eval()
    distiller = make_distiller(teacher, student, L=1, M=1)
    (loss,) = distiller(**batch)
    assert torch.isfinite(loss) and loss.item() > 0
    loss.backward()


def test_distiller_length_one_batch(teacher, student):
    input_ids = torch.tensor([[5], [7]])
    mask = torch.ones(2, 1, dtype=torch.long)
    distiller = make_distiller(teacher, student)
    (loss,) = distiller(input_ids=input_ids, attention_mask=mask)
    assert torch.isfinite(loss)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_distiller_batch_with_all_pad_row(teacher, student, batch):
    """An all-pad row in a real batch (e.g. last partial batch from a packer)
    must neither NaN the loss nor poison the gradients."""
    input_ids = torch.cat([batch["input_ids"],
                           torch.full((1, 8), PAD_ID, dtype=torch.long)])
    mask = torch.cat([batch["attention_mask"], torch.zeros(1, 8, dtype=torch.long)])
    distiller = make_distiller(teacher, student)
    (loss,) = distiller(input_ids=input_ids, attention_mask=mask)
    assert torch.isfinite(loss)
    loss.backward()
    for p in student.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all()


def test_forward_twice_then_backward_both(teacher, student, batch):
    """Two forwards before any backward (e.g. gradient accumulation) must keep
    two independent graphs; backward on both must accumulate finite grads."""
    distiller = make_distiller(teacher, student)
    (loss1,) = distiller(**batch)
    (loss2,) = distiller(**batch)
    torch.testing.assert_close(loss1, loss2)  # eval mode, same inputs
    loss1.backward()
    loss2.backward()  # would raise if graphs/buffers were shared incorrectly
    grads = [p.grad for p in student.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_state_dict_round_trip(teacher, student, batch):
    distiller = make_distiller(teacher, student)
    (base,) = distiller(**batch)
    state = {k: v.clone() for k, v in distiller.state_dict().items()}

    torch.manual_seed(0)
    teacher2 = tiny_bert(hidden_size=32, num_layers=3, num_heads=4).eval()
    torch.manual_seed(99)  # deliberately different student weights
    student2 = tiny_bert(hidden_size=16, num_layers=2, num_heads=2).eval()
    distiller2 = make_distiller(teacher2, student2)
    distiller2.load_state_dict(state)
    (restored,) = distiller2(**batch)
    torch.testing.assert_close(restored, base)


def test_train_mode_does_not_unfreeze_teacher_dropout(teacher, student):
    distiller = make_distiller(teacher, student)
    distiller.train()
    assert distiller.student.training
    assert not distiller.teacher.training


def test_eval_train_toggle_propagates_to_student(teacher, student):
    distiller = make_distiller(teacher, student)
    distiller.train()
    assert distiller.student.training
    distiller.eval()
    assert not distiller.student.training and not distiller.teacher.training


def test_relations_dict_is_copied(teacher, student, batch):
    """Mutating the caller's relations dict after construction must not change
    the distiller's behavior."""
    relations = {(1, 1): 1.0}
    distiller = make_distiller(teacher, student, relations=relations)
    (base,) = distiller(**batch)
    relations[(2, 2)] = 100.0
    (after,) = distiller(**batch)
    torch.testing.assert_close(after, base)


# ---------------------------------------------------------------------------
# Data pipeline
# ---------------------------------------------------------------------------

def test_tokenize_fn_pickle_round_trip(tmp_path):
    """_TokenizeFn must survive pickling (DataLoader workers pickle the
    transform) and behave identically afterwards."""
    from transformers import BertTokenizer

    from train.data_pipeline import _TokenizeFn

    vocab = tmp_path / "vocab.txt"
    vocab.write_text("\n".join(
        ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "hello", "world", "##s"]
    ))
    tokenizer = BertTokenizer(str(vocab))
    fn = _TokenizeFn(tokenizer, text_column="text", max_seq_len=8)

    examples = {"text": ["hello world", "world hello hello"]}
    before = fn(examples)
    restored = pickle.loads(pickle.dumps(fn))
    after = restored(examples)
    assert before["input_ids"] == after["input_ids"]
    assert restored.max_seq_len == 8 and restored.text_column == "text"
    # truncation must actually cap at max_seq_len
    long = fn({"text": ["hello " * 50]})
    assert len(long["input_ids"][0]) == 8
