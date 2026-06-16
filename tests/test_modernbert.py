import pytest
import torch
from transformers import ModernBertConfig, ModernBertModel

from train.adapters import ModernBertRecorder, create_recorder
from train.distiller import MiniLMDistiller

RELATIONS = {(1, 1): 1.0, (2, 2): 1.0, (3, 3): 1.0}
PAD_ID = 0


def tiny_modernbert(num_layers=4):
    config = ModernBertConfig(
        vocab_size=99, hidden_size=32, num_hidden_layers=num_layers,
        num_attention_heads=4, intermediate_size=64, max_position_embeddings=64,
        global_attn_every_n_layers=3, local_attention=8,
        pad_token_id=PAD_ID, bos_token_id=1, eos_token_id=2,
        cls_token_id=1, sep_token_id=2,
    )
    torch.manual_seed(0)
    return ModernBertModel(config).eval()


@pytest.fixture
def modernbert():
    return tiny_modernbert()


@pytest.fixture
def batch():
    torch.manual_seed(2)
    input_ids = torch.randint(3, 99, (3, 8))
    mask = torch.zeros(3, 8, dtype=torch.long)
    for row, length in enumerate((8, 5, 3)):
        mask[row, :length] = 1
        input_ids[row, length:] = PAD_ID
    return {"input_ids": input_ids, "attention_mask": mask}


def test_registry_dispatch(modernbert):
    recorder = create_recorder(modernbert, 1)
    assert isinstance(recorder, ModernBertRecorder)
    assert not recorder.accepts_token_type_ids
    recorder.close()


def test_sliding_window_layer_rejected(modernbert):
    # layer_types: [full, sliding, sliding, full] for global_attn_every_n_layers=3
    with pytest.raises(ValueError, match="sliding-window"):
        ModernBertRecorder(modernbert, 2)
    ModernBertRecorder(modernbert, 4).close()  # last layer is global


def test_hooked_qkv_equals_manual_recompute(modernbert, batch):
    L = 4
    recorder = ModernBertRecorder(modernbert, L)
    with torch.no_grad():
        out = modernbert(**batch, output_hidden_states=True)
    q, k, v = recorder.pop()

    layer = modernbert.layers[L - 1]
    with torch.no_grad():
        qkv = layer.attn.Wqkv(layer.attn_norm(out.hidden_states[L - 1]))
    B, S, _ = qkv.shape
    q_ref, k_ref, v_ref = (
        t.reshape(B, S, 32) for t in qkv.view(B, S, 3, 4, 8).unbind(dim=-3)
    )
    torch.testing.assert_close(q, q_ref)
    torch.testing.assert_close(k, k_ref)
    torch.testing.assert_close(v, v_ref)
    recorder.close()


def test_distillation_end_to_end(modernbert, batch):
    from conftest import tiny_bert

    torch.manual_seed(1)
    student = tiny_bert(hidden_size=16, num_layers=2, num_heads=2).eval()
    distiller = MiniLMDistiller(
        teacher=modernbert, student=student, L=4, M=2,
        relations=RELATIONS, A_r=4, teacher_dtype=torch.float32,
    )
    assert len(modernbert.layers) == 4  # truncation keeps all 4 (L == depth)

    # token_type_ids must be withheld from ModernBERT but still reach the student
    (loss,) = distiller(**batch, token_type_ids=torch.zeros_like(batch["input_ids"]))
    assert torch.isfinite(loss) and loss.item() > 0
    loss.backward()
    assert any(p.grad is not None for p in student.parameters())

    # pad invariance with ModernBERT's pad handling
    scrambled = batch["input_ids"].clone()
    pad = batch["attention_mask"] == 0
    scrambled[pad] = torch.randint(3, 99, (int(pad.sum()),))
    (loss2,) = distiller(input_ids=scrambled, attention_mask=batch["attention_mask"])
    (loss1,) = distiller(**batch)
    torch.testing.assert_close(loss2, loss1, rtol=1e-5, atol=1e-6)


def tiny_modernbert_student(num_layers=2):
    config = ModernBertConfig(
        vocab_size=99, hidden_size=16, num_hidden_layers=num_layers,
        num_attention_heads=2, intermediate_size=32, max_position_embeddings=64,
        local_attention=8,
        pad_token_id=PAD_ID, bos_token_id=1, eos_token_id=2,
        cls_token_id=1, sep_token_id=2,
    )
    # all-global layers must be set post-init: uniform layer_types in the
    # constructor trips ModernBertConfig's strict rope_parameters validation
    config.global_attn_every_n_layers = 1
    config.layer_types = ["full_attention"] * num_layers
    torch.manual_seed(5)
    return ModernBertModel(config)


def test_modernbert_student_gets_gradients(modernbert, batch):
    """ModernBERT as the *student*: hooks must stay in the autograd graph."""
    student = tiny_modernbert_student()
    distiller = MiniLMDistiller(
        teacher=modernbert, student=student, L=4, M=2,
        relations=RELATIONS, A_r=4, teacher_dtype=torch.float32,
    )
    (loss,) = distiller(**batch)
    loss.backward()
    wqkv_grad = student.layers[1].attn.Wqkv.weight.grad
    assert wqkv_grad is not None and wqkv_grad.abs().sum() > 0


def test_bert_teacher_modernbert_student(batch):
    from conftest import tiny_bert

    torch.manual_seed(0)
    teacher = tiny_bert(hidden_size=32, num_layers=3, num_heads=4).eval()
    student = tiny_modernbert_student()
    distiller = MiniLMDistiller(
        teacher=teacher, student=student, L=2, M=2,
        relations=RELATIONS, A_r=4, teacher_dtype=torch.float32,
    )
    (loss,) = distiller(**batch)
    assert torch.isfinite(loss) and loss.item() > 0
    loss.backward()


def test_packed_forward_equals_isolated_forwards(modernbert):
    """Block ∧ sliding mask composition: every doc in a packed row must get
    bit-equal hidden states to running it alone - including a doc longer than
    the sliding window, which exercises the sliding_attention layers."""
    from train.adapters import ModernBertRecorder

    recorder = ModernBertRecorder(modernbert, 4)
    torch.manual_seed(3)
    doc_a = torch.randint(3, 99, (1, 12))  # longer than local_attention=8
    doc_b = torch.randint(3, 99, (1, 6))
    packed = torch.cat([doc_a, doc_b], dim=1)
    seg = torch.tensor([[0] * 12 + [1] * 6])
    pos = torch.tensor([[*range(12), *range(6)]])
    mapping = recorder.build_packed_attention(torch.ones(1, 18, dtype=torch.long), seg)

    with torch.no_grad():
        out_packed = modernbert(input_ids=packed, attention_mask=mapping, position_ids=pos).last_hidden_state
        out_a = modernbert(input_ids=doc_a).last_hidden_state
        out_b = modernbert(input_ids=doc_b).last_hidden_state
    torch.testing.assert_close(out_packed[0, :12], out_a[0], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(out_packed[0, 12:], out_b[0], rtol=1e-5, atol=1e-5)
    recorder.close()


def test_packed_distillation_equals_separate(modernbert):
    """End-to-end packed mmBERT-style distillation must equal the
    token-weighted combination of per-document losses, with padding."""
    student = tiny_modernbert_student()
    distiller = MiniLMDistiller(
        teacher=modernbert, student=student, L=4, M=2,
        relations=RELATIONS, A_r=4, teacher_dtype=torch.float32,
    )
    torch.manual_seed(4)
    l1, l2, S = 10, 5, 18
    doc1 = torch.randint(3, 99, (1, l1))
    doc2 = torch.randint(3, 99, (1, l2))
    packed_ids = torch.cat([doc1, doc2, torch.zeros(1, S - l1 - l2, dtype=torch.long)], dim=1)
    mask = torch.tensor([[1] * (l1 + l2) + [0] * (S - l1 - l2)])
    seg = torch.tensor([[0] * l1 + [1] * l2 + [-1] * (S - l1 - l2)])
    pos = torch.tensor([[*range(l1), *range(l2)] + [0] * (S - l1 - l2)])

    (packed,) = distiller(input_ids=packed_ids, attention_mask=mask, segment_ids=seg, position_ids=pos)
    (loss1,) = distiller(input_ids=doc1, attention_mask=torch.ones(1, l1, dtype=torch.long))
    (loss2,) = distiller(input_ids=doc2, attention_mask=torch.ones(1, l2, dtype=torch.long))

    expected = (loss1 * 4 * l1 + loss2 * 4 * l2) / (4 * (l1 + l2))
    torch.testing.assert_close(packed, expected, rtol=1e-4, atol=1e-5)


def test_packed_inputs_rejected_for_ltg():
    from test_adapters import FakeLtgModel

    from train.adapters import create_recorder

    model = FakeLtgModel()
    recorder = create_recorder(model, 1)
    assert not recorder.supports_packing
    recorder.close()


def test_truncation(modernbert, batch):
    distiller_loss = None
    torch.manual_seed(1)
    from conftest import tiny_bert

    for truncate in (False, True):
        teacher = tiny_modernbert()
        torch.manual_seed(1)
        student = tiny_bert(hidden_size=16, num_layers=2, num_heads=2).eval()
        d = MiniLMDistiller(
            teacher=teacher, student=student, L=1, M=2,
            relations=RELATIONS, A_r=4, teacher_dtype=torch.float32,
            truncate_teacher=truncate,
        )
        (loss,) = d(**batch)
        if truncate:
            assert len(teacher.layers) == 1
            torch.testing.assert_close(loss, distiller_loss)
        distiller_loss = loss
