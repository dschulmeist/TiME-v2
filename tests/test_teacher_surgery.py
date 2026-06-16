"""Tests for two just-added features:

1. Surgical distill-layer stub (ModernBERT): truncate_layers() replaces the
   hooked layer with _ModernBertWqkvOnly (attn_norm + Wqkv only).
2. MiniLMDistiller(compile_teacher=True): in-place torch.compile of the teacher.
"""
import pytest
import torch
from transformers import ModernBertConfig, ModernBertModel

from train.adapters import (
    BertLikeRecorder,
    ModernBertRecorder,
    _ModernBertWqkvOnly,
    create_recorder,
)
from train.distiller import MiniLMDistiller

RELATIONS = {(1, 1): 1.0, (2, 2): 1.0, (3, 3): 1.0}
PAD_ID = 0


def tiny_modernbert(num_layers=4, seed=0):
    config = ModernBertConfig(
        vocab_size=99, hidden_size=32, num_hidden_layers=num_layers,
        num_attention_heads=4, intermediate_size=64, max_position_embeddings=64,
        global_attn_every_n_layers=3, local_attention=8,
        pad_token_id=PAD_ID, bos_token_id=1, eos_token_id=2,
        cls_token_id=1, sep_token_id=2,
    )
    torch.manual_seed(seed)
    return ModernBertModel(config).eval()


def tiny_student():
    from conftest import tiny_bert

    torch.manual_seed(1)
    return tiny_bert(hidden_size=16, num_layers=2, num_heads=2).eval()


@pytest.fixture
def batch():
    torch.manual_seed(2)
    input_ids = torch.randint(3, 99, (3, 8))
    mask = torch.zeros(3, 8, dtype=torch.long)
    for row, length in enumerate((8, 5, 3)):
        mask[row, :length] = 1
        input_ids[row, length:] = PAD_ID
    return {"input_ids": input_ids, "attention_mask": mask}


def recorded_qkv(model, L, batch, truncate):
    """Hook layer L, optionally truncate surgically, run forward, return Q/K/V."""
    recorder = ModernBertRecorder(model, L)
    if truncate:
        recorder.truncate_layers()
    with torch.no_grad():
        model(**batch)
    qkv = recorder.pop()
    recorder.close()
    return qkv


# -----------------------------------------------------------------------------
# Feature 1: surgical distill-layer stub
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("num_layers,L", [(6, 4), (4, 1), (4, 4)])
def test_surgical_qkv_identical_to_full_model(batch, num_layers, L):
    """Q/K/V recorded through the Wqkv-only stub must be bit-identical to the
    full untruncated model - mid-stack L, L == 1, and L == depth."""
    full = tiny_modernbert(num_layers)
    surgical = tiny_modernbert(num_layers)  # same seed -> identical weights
    q0, k0, v0 = recorded_qkv(full, L, batch, truncate=False)
    q1, k1, v1 = recorded_qkv(surgical, L, batch, truncate=True)
    torch.testing.assert_close(q1, q0, rtol=0, atol=0)
    torch.testing.assert_close(k1, k0, rtol=0, atol=0)
    torch.testing.assert_close(v1, v0, rtol=0, atol=0)


def test_surgery_state(batch):
    """After surgery the model has exactly L layers, the last is the stub,
    and config.num_hidden_layers is updated."""
    model = tiny_modernbert(6)
    orig_layer = model.layers[3]
    orig_wqkv = orig_layer.attn.Wqkv
    recorder = ModernBertRecorder(model, 4)
    recorder.truncate_layers()
    assert len(model.layers) == 4
    assert model.config.num_hidden_layers == 4
    assert isinstance(model.layers[3], _ModernBertWqkvOnly)
    # the stub reuses the original submodule objects (hooks stay attached)
    assert model.layers[3].Wqkv is orig_wqkv
    assert model.layers[3].attn_norm is orig_layer.attn_norm
    # attention_type must be preserved for the per-layer-type mask dict path
    assert model.layers[3].attention_type == orig_layer.attention_type == "full_attention"
    # earlier layers untouched
    assert not isinstance(model.layers[2], _ModernBertWqkvOnly)
    recorder.close()


def test_hooks_registered_before_surgery_still_fire(batch):
    """The recorder attaches hooks in __init__; truncate_layers() must keep
    them firing (the stub reuses the hooked Wqkv module object)."""
    model = tiny_modernbert(4)
    recorder = ModernBertRecorder(model, 4)
    recorder.truncate_layers()
    with torch.no_grad():
        model(**batch)
    q, k, v = recorder.pop()  # raises if the hook did not fire
    assert q.shape == k.shape == v.shape == (3, 8, 32)
    # cache cleared by pop; a second forward must refill it
    with torch.no_grad():
        model(**batch)
    q2, _, _ = recorder.pop()
    torch.testing.assert_close(q2, q, rtol=0, atol=0)
    recorder.close()


def test_stub_passes_hidden_states_through_unchanged(batch):
    model = tiny_modernbert(4)
    recorder = ModernBertRecorder(model, 4)
    recorder.truncate_layers()
    stub = model.layers[3]
    torch.manual_seed(7)
    x = torch.randn(2, 5, 32)
    with torch.no_grad():
        out = stub(x)
    # the stub returns its input unchanged (its outputs are never consumed)
    assert out is x
    # extra positional/keyword encoder arguments must be tolerated
    with torch.no_grad():
        out2 = stub(x, None, sliding_window_mask=None, position_ids=None)
    assert out2 is x
    recorder.close()


@pytest.mark.parametrize("num_layers,L", [(6, 4), (4, 1)])
def test_distiller_loss_surgical_vs_full(batch, num_layers, L):
    """End-to-end distiller loss identical with and without teacher surgery."""
    losses = {}
    for truncate in (False, True):
        teacher = tiny_modernbert(num_layers)
        d = MiniLMDistiller(
            teacher=teacher, student=tiny_student(), L=L, M=2,
            relations=RELATIONS, A_r=4, teacher_dtype=torch.float32,
            truncate_teacher=truncate,
        )
        (losses[truncate],) = d(**batch)
        if truncate:
            assert len(teacher.layers) == L
            assert isinstance(teacher.layers[L - 1], _ModernBertWqkvOnly)
    torch.testing.assert_close(losses[True], losses[False], rtol=0, atol=0)
    assert torch.isfinite(losses[True]) and losses[True].item() > 0


def test_packed_through_surgical_teacher_equals_isolated():
    """Packed inputs through a surgically truncated mid-stack teacher must
    yield the same recorded Q/K/V as isolated per-document forwards. With
    L=4 of 6, layers 2-3 use sliding attention, so the mask dict's
    per-attention_type selection (which the stub's attention_type attribute
    feeds) is exercised, including a doc longer than the window."""
    model = tiny_modernbert(6)
    recorder = ModernBertRecorder(model, 4)
    recorder.truncate_layers()

    torch.manual_seed(3)
    doc_a = torch.randint(3, 99, (1, 12))  # > local_attention=8
    doc_b = torch.randint(3, 99, (1, 6))
    packed = torch.cat([doc_a, doc_b], dim=1)
    seg = torch.tensor([[0] * 12 + [1] * 6])
    pos = torch.tensor([[*range(12), *range(6)]])
    mapping = recorder.build_packed_attention(torch.ones(1, 18, dtype=torch.long), seg)

    with torch.no_grad():
        model(input_ids=packed, attention_mask=mapping, position_ids=pos)
        q_p, k_p, v_p = recorder.pop()
        model(input_ids=doc_a)
        q_a, k_a, v_a = recorder.pop()
        model(input_ids=doc_b)
        q_b, k_b, v_b = recorder.pop()

    for packed_t, a, b in ((q_p, q_a, q_b), (k_p, k_a, k_b), (v_p, v_a, v_b)):
        torch.testing.assert_close(packed_t[0, :12], a[0], rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(packed_t[0, 12:], b[0], rtol=1e-5, atol=1e-5)
    recorder.close()


def test_bert_family_keeps_full_layer(batch):
    """BERT recorders return a None stub: the hooked layer stays a full
    BertLayer and truncation behaves exactly as before."""
    from transformers.models.bert.modeling_bert import BertLayer

    from conftest import tiny_bert

    torch.manual_seed(0)
    full = tiny_bert(hidden_size=32, num_layers=3, num_heads=4).eval()
    torch.manual_seed(0)
    truncated = tiny_bert(hidden_size=32, num_layers=3, num_heads=4).eval()

    rec = BertLikeRecorder(truncated, 2)
    assert rec._distill_layer_stub(truncated.encoder.layer[1]) is None
    rec.truncate_layers()
    assert len(truncated.encoder.layer) == 2
    assert truncated.config.num_hidden_layers == 2
    assert isinstance(truncated.encoder.layer[1], BertLayer)

    rec_full = BertLikeRecorder(full, 2)
    with torch.no_grad():
        full(**batch)
        truncated(**batch)
    qkv_full = rec_full.pop()
    qkv_trunc = rec.pop()
    for a, b in zip(qkv_trunc, qkv_full):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    rec.close()
    rec_full.close()


def test_stub_follows_dtype_cast(batch):
    """teacher.to(dtype) before surgery: the stub's reused modules must carry
    the cast, and the recorded tensors come out in the teacher dtype."""
    teacher = tiny_modernbert(4)
    d = MiniLMDistiller(
        teacher=teacher, student=tiny_student(), L=4, M=2,
        relations=RELATIONS, A_r=4, teacher_dtype=torch.float64,
    )
    assert teacher.layers[3].Wqkv.weight.dtype == torch.float64
    (loss,) = d(**batch)
    assert torch.isfinite(loss)


def test_create_recorder_dispatch_then_surgery(batch):
    """create_recorder path (as used by the distiller) yields a recorder whose
    surgery works identically to direct construction."""
    model = tiny_modernbert(4)
    recorder = create_recorder(model, 4)
    assert isinstance(recorder, ModernBertRecorder)
    recorder.truncate_layers()
    with torch.no_grad():
        model(**batch)
    q, k, v = recorder.pop()
    assert q.shape == (3, 8, 32)
    recorder.close()


# -----------------------------------------------------------------------------
# Feature 2: compile_teacher
# -----------------------------------------------------------------------------

def test_compile_teacher_smoke(batch):
    """compile_teacher=True must produce the same loss as eager (rtol 1e-4)
    and backward through the student must still work."""
    pytest.importorskip("torch._dynamo")

    teacher_eager = tiny_modernbert(4)
    d_eager = MiniLMDistiller(
        teacher=teacher_eager, student=tiny_student(), L=4, M=2,
        relations=RELATIONS, A_r=4, teacher_dtype=torch.float32,
    )
    (loss_eager,) = d_eager(**batch)

    teacher_c = tiny_modernbert(4)
    student_c = tiny_student()
    d_compiled = MiniLMDistiller(
        teacher=teacher_c, student=student_c, L=4, M=2,
        relations=RELATIONS, A_r=4, teacher_dtype=torch.float32,
        compile_teacher=True,
    )
    # compile() is in-place: the distiller still holds the same module object
    assert d_compiled.teacher is teacher_c
    (loss_c,) = d_compiled(**batch)
    torch.testing.assert_close(loss_c, loss_eager, rtol=1e-4, atol=1e-6)
    loss_c.backward()
    grads = [p.grad for p in student_c.parameters() if p.grad is not None]
    assert grads and any(g.abs().sum() > 0 for g in grads)
