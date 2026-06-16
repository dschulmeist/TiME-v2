import pytest
import torch
from transformers import BertConfig, BertForMaskedLM

from train.distiller import MiniLMDistiller
from train.losses import hidden_state_mse, logit_kd

from conftest import tiny_bert

RELATIONS = {(1, 1): 1.0, (2, 2): 1.0, (3, 3): 1.0}


# --- loss-level ---------------------------------------------------------------

def ragged_mask(B=3, S=8, lengths=(8, 5, 3)):
    m = torch.zeros(B, S, dtype=torch.long)
    for r, n in enumerate(lengths):
        m[r, :n] = 1
    return m


def test_hidden_state_mse_zero_when_identical():
    h = torch.randn(3, 8, 16)
    loss = hidden_state_mse(h, h, ragged_mask(), projection=None)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_hidden_state_mse_ignores_padding():
    torch.manual_seed(0)
    ht, hs = torch.randn(3, 8, 16), torch.randn(3, 8, 16)
    mask = ragged_mask()
    base = hidden_state_mse(ht, hs, mask, projection=None)
    pad = (mask == 0)[..., None]
    scrambled = torch.where(pad, torch.randn_like(hs) * 10, hs)
    assert torch.allclose(hidden_state_mse(ht, scrambled, mask, projection=None), base, atol=1e-6)


def test_hidden_state_mse_projection_grad():
    proj = torch.nn.Linear(8, 16)
    ht, hs = torch.randn(2, 5, 16), torch.randn(2, 5, 8)
    loss = hidden_state_mse(ht, hs, torch.ones(2, 5, dtype=torch.long), projection=proj)
    loss.backward()
    assert proj.weight.grad is not None and proj.weight.grad.abs().sum() > 0


def test_logit_kd_zero_when_identical_and_temperature_scales():
    torch.manual_seed(0)
    logits = torch.randn(2, 6, 50)
    mask = torch.ones(2, 6, dtype=torch.long)
    assert logit_kd(logits, logits, mask).item() == pytest.approx(0.0, abs=1e-6)
    # teacher != student: finite, positive, differentiable
    s = torch.randn(2, 6, 50, requires_grad=True)
    loss = logit_kd(torch.randn(2, 6, 50), s, mask, temperature=2.0)
    assert loss.item() > 0
    loss.backward()
    assert torch.isfinite(s.grad).all()


def test_logit_kd_seq_chunk_equals_unchunked():
    """Sequence-chunking (memory bound for big vocab) must not change the loss."""
    torch.manual_seed(2)
    t = torch.randn(3, 10, 60)
    s = torch.randn(3, 10, 60, requires_grad=True)
    mask = torch.ones(3, 10, dtype=torch.long); mask[2, 6:] = 0
    full = logit_kd(t, s, mask, temperature=2.0, seq_chunk_size=None)
    for chunk in (1, 3, 4, 10, 99):
        torch.testing.assert_close(logit_kd(t, s, mask, 2.0, seq_chunk_size=chunk), full,
                                   rtol=1e-5, atol=1e-6)


def test_logit_kd_ignores_padding():
    torch.manual_seed(1)
    t, s = torch.randn(2, 6, 40), torch.randn(2, 6, 40)
    mask = torch.tensor([[1, 1, 1, 1, 0, 0], [1, 1, 1, 0, 0, 0]])
    base = logit_kd(t, s, mask)
    s2 = s.clone(); s2[mask == 0] = torch.randn_like(s2[mask == 0]) * 10
    assert torch.allclose(logit_kd(t, s2, mask), base, atol=1e-6)


# --- distiller-level ----------------------------------------------------------

def tiny_mlm(vocab=99, hidden=32, layers=3, heads=4):
    cfg = BertConfig(vocab_size=vocab, hidden_size=hidden, num_hidden_layers=layers,
                     num_attention_heads=heads, intermediate_size=hidden * 4,
                     max_position_embeddings=64, pad_token_id=0)
    torch.manual_seed(0)
    return BertForMaskedLM(cfg).eval()


def test_repr_term_builds_projection_and_flows(teacher, student, batch):
    d = MiniLMDistiller(teacher=teacher, student=student, L=2, M=2, relations=RELATIONS,
                        A_r=4, teacher_dtype=torch.float32, repr_weight=1.0)
    assert d.hidden_projection is not None
    # capture_hidden keeps the full layer L (no surgical stub / truncation collapse)
    (loss,) = d(**batch)
    assert torch.isfinite(loss) and loss.item() > 0
    loss.backward()
    assert d.hidden_projection.weight.grad is not None
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in student.parameters())


def test_weight_zero_disables_terms(teacher, student, batch):
    """repr_weight=0 / logit_kd_weight=0 must add no term and no aux params."""
    d = MiniLMDistiller(teacher=teacher, student=student, L=2, M=2, relations=RELATIONS,
                        A_r=4, teacher_dtype=torch.float32, repr_weight=0.0, logit_kd_weight=0.0)
    assert d.hidden_projection is None
    assert not d._needs_hidden and not d._needs_logit
    # equals a plain relation-only distiller on the same objects
    (with_zero,) = d(**batch)
    plain = MiniLMDistiller(teacher=teacher, student=student, L=2, M=2, relations=RELATIONS,
                            A_r=4, teacher_dtype=torch.float32)
    (relation_only,) = plain(**batch)
    torch.testing.assert_close(with_zero, relation_only)


def test_logit_kd_end_to_end_and_no_truncation():
    teacher, student = tiny_mlm(), tiny_mlm(hidden=16, layers=2, heads=2)
    d = MiniLMDistiller(teacher=teacher, student=student, L=2, M=2, relations=RELATIONS,
                        A_r=4, teacher_dtype=torch.float32, logit_kd_weight=1.0)
    # logit-KD must keep the full teacher (all 3 layers), not truncate to L=2
    assert len(teacher.bert.encoder.layer) == 3
    ids = torch.randint(1, 99, (2, 8)); mask = torch.ones(2, 8, dtype=torch.long)
    (loss,) = d(input_ids=ids, attention_mask=mask)
    assert torch.isfinite(loss) and loss.item() > 0
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in student.parameters())


def test_logit_kd_rejects_vocab_mismatch():
    teacher, student = tiny_mlm(vocab=99), tiny_mlm(vocab=50, hidden=16, layers=2, heads=2)
    with pytest.raises(ValueError, match="shared vocabulary"):
        MiniLMDistiller(teacher=teacher, student=student, L=2, M=2, relations=RELATIONS,
                        A_r=4, teacher_dtype=torch.float32, logit_kd_weight=1.0)


def test_logit_kd_rejects_vocab_pruning():
    teacher, student = tiny_mlm(), tiny_mlm(hidden=16, layers=2, heads=2)
    remap = torch.arange(99)
    with pytest.raises(ValueError, match="prune"):
        MiniLMDistiller(teacher=teacher, student=student, L=2, M=2, relations=RELATIONS,
                        A_r=4, teacher_dtype=torch.float32, logit_kd_weight=1.0,
                        student_input_remap=remap)


def test_state_dict_roundtrip_drops_teacher_and_guards_trainables(teacher, student):
    d = MiniLMDistiller(teacher=teacher, student=student, L=2, M=2, relations=RELATIONS,
                        A_r=4, teacher_dtype=torch.float32, repr_weight=1.0)
    sd = d.state_dict()
    # teacher is dropped; student + projection are kept
    assert not any(k.startswith("teacher.") for k in sd)
    assert any(k.startswith("student.") for k in sd)
    assert any(k.startswith("hidden_projection.") for k in sd)
    # a teacher-less checkpoint loads cleanly (teacher rebuilt separately)
    d.load_state_dict(sd)
    # but a genuinely missing trainable weight must raise, not pass silently
    broken = {k: v for k, v in sd.items() if "hidden_projection" not in k}
    with pytest.raises(RuntimeError, match="missing trainable keys"):
        d.load_state_dict(broken)


def test_all_terms_compose(teacher, student, batch):
    """relation + repr together stay finite with grads to all trainables."""
    d = MiniLMDistiller(teacher=teacher, student=student, L=2, M=2, relations=RELATIONS,
                        A_r=4, teacher_dtype=torch.float32, repr_weight=0.5)
    (loss,) = d(**batch)
    loss.backward()
    assert torch.isfinite(loss)
    assert d.hidden_projection.weight.grad is not None
