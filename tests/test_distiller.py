import pytest
import torch

from train.distiller import MiniLMDistiller

RELATIONS = {(1, 1): 1.0, (2, 2): 1.0, (3, 3): 1.0}


def make_distiller(teacher, student, **kwargs):
    defaults = dict(
        teacher=teacher, student=student, L=2, M=2,
        relations=RELATIONS, A_r=4, teacher_dtype=torch.float32,
    )
    defaults.update(kwargs)
    return MiniLMDistiller(**defaults)


def test_forward_returns_finite_loss(teacher, student, batch):
    distiller = make_distiller(teacher, student)
    (loss,) = distiller(**batch)
    assert loss.dtype == torch.float32
    assert torch.isfinite(loss) and loss.item() > 0


def test_teacher_frozen_student_gets_gradients(teacher, student, batch):
    distiller = make_distiller(teacher, student)
    (loss,) = distiller(**batch)
    loss.backward()
    assert all(not p.requires_grad for p in teacher.parameters())
    grads = [p.grad for p in student.parameters() if p.grad is not None]
    assert grads and any(g.abs().sum() > 0 for g in grads)


def test_truncation_does_not_change_loss(teacher, student, batch):
    torch.manual_seed(0)
    full = make_distiller(teacher, student, truncate_teacher=False)
    (loss_full,) = full(**batch)

    (loss_truncated,) = make_distiller(teacher, student)(**batch)
    assert len(teacher.encoder.layer) == 2
    torch.testing.assert_close(loss_truncated, loss_full)


def test_pad_invariance_end_to_end(teacher, student, batch):
    """Changing token ids at padded positions must not change the loss."""
    distiller = make_distiller(teacher, student)
    (base,) = distiller(**batch)

    scrambled = batch["input_ids"].clone()
    pad = batch["attention_mask"] == 0
    scrambled[pad] = torch.randint(1, 99, (int(pad.sum()),))
    (loss,) = distiller(input_ids=scrambled, attention_mask=batch["attention_mask"])
    torch.testing.assert_close(loss, base, rtol=1e-5, atol=1e-6)


def test_attention_mask_fallback_uses_pad_token_id(teacher, student, batch):
    distiller = make_distiller(teacher, student)
    (explicit,) = distiller(**batch)
    # conftest pads with the configured pad_token_id, so the inferred mask matches
    (inferred,) = distiller(input_ids=batch["input_ids"])
    torch.testing.assert_close(inferred, explicit)


def test_missing_pad_token_id_raises(teacher, student, batch):
    distiller = make_distiller(teacher, student)
    distiller._pad_token_id = None
    with pytest.raises(ValueError, match="attention_mask"):
        distiller(input_ids=batch["input_ids"])


def test_bf16_teacher(teacher, student, batch):
    distiller = make_distiller(teacher, student, teacher_dtype=torch.bfloat16)
    assert next(teacher.parameters()).dtype == torch.bfloat16
    (loss,) = distiller(**batch)
    assert torch.isfinite(loss)
    loss.backward()


def test_invalid_relation_heads_raise(teacher, student):
    with pytest.raises(ValueError, match="A_r"):
        make_distiller(teacher, student, A_r=5)


def test_chunked_equals_unchunked_end_to_end(teacher, student, batch):
    torch.manual_seed(0)
    (chunked,) = make_distiller(teacher, student, head_chunk_size=1)(**batch)
    (full,) = make_distiller(teacher, student, head_chunk_size=None)(**batch)
    torch.testing.assert_close(chunked, full, rtol=1e-6, atol=1e-7)


def test_loss_is_fp32_under_autocast(teacher, student, batch):
    distiller = make_distiller(teacher, student)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        (loss,) = distiller(**batch)
    assert loss.dtype == torch.float32 and torch.isfinite(loss)


def test_gradient_checkpointing_delegates_to_student(teacher, student, batch):
    distiller = make_distiller(teacher, student)
    distiller.gradient_checkpointing_enable()
    assert student.is_gradient_checkpointing
    student.train()
    (loss,) = distiller(**batch)
    loss.backward()
    assert any(p.grad is not None for p in student.parameters())
    distiller.gradient_checkpointing_disable()
    assert not student.is_gradient_checkpointing


def test_token_type_ids_forwarded_only_when_supported(teacher, student, batch):
    distiller = make_distiller(teacher, student)
    token_type_ids = torch.zeros_like(batch["input_ids"])
    (with_tti,) = distiller(**batch, token_type_ids=token_type_ids)

    # a teacher that rejects token_type_ids (e.g. ModernBERT) must not receive them
    distiller.teacher_recorder.accepts_token_type_ids = False
    seen = {}
    original_forward = teacher.forward

    def spy(*args, **kwargs):
        seen.update(kwargs)
        return original_forward(*args, **kwargs)

    teacher.forward = spy
    distiller(**batch, token_type_ids=token_type_ids)
    assert "token_type_ids" not in seen
    teacher.forward = original_forward
    assert torch.isfinite(with_tti)


def test_extra_collator_columns_ignored(teacher, student, batch):
    """The Trainer may pass extra keys (labels, special_tokens_mask, ...)."""
    distiller = make_distiller(teacher, student)
    (loss,) = distiller(**batch, special_tokens_mask=torch.zeros_like(batch["input_ids"]))
    assert torch.isfinite(loss)


def test_one_optimizer_step_reduces_loss(teacher, student, batch):
    torch.manual_seed(0)
    distiller = make_distiller(teacher, student)
    optimizer = torch.optim.AdamW(student.parameters(), lr=5e-4)
    (first,) = distiller(**batch)
    first.backward()
    optimizer.step()
    optimizer.zero_grad()
    (second,) = distiller(**batch)
    assert second.item() < first.item()
