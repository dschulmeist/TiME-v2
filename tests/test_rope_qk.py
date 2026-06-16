import pytest
import torch
from transformers import ModernBertConfig, ModernBertModel
from transformers.models.modernbert.modeling_modernbert import apply_rotary_pos_emb

from train.adapters import ModernBertRecorder, create_recorder


def tiny_modernbert(num_layers=4):
    cfg = ModernBertConfig(
        vocab_size=99, hidden_size=32, num_hidden_layers=num_layers,
        num_attention_heads=4, intermediate_size=64, max_position_embeddings=64,
        global_attn_every_n_layers=3, local_attention=8,
        pad_token_id=0, bos_token_id=1, eos_token_id=2, cls_token_id=1, sep_token_id=2,
        reference_compile=False,
    )
    torch.manual_seed(0)
    return ModernBertModel(cfg).eval()


def model_real_rope_qk(model, layer, input_ids):
    """The post-RoPE q,k the model actually computes at `layer`, captured by
    replicating its forward from the Wqkv output + the layer's cos/sin."""
    attn = model.layers[layer - 1].attn
    cap = {}
    h1 = attn.register_forward_pre_hook(
        lambda m, a, kw: cap.__setitem__("pos", kw["position_embeddings"]), with_kwargs=True)
    h2 = attn.Wqkv.register_forward_hook(lambda m, i, o: cap.__setitem__("qkv", o.detach()))
    with torch.no_grad():
        model(input_ids=input_ids)
    h1.remove(); h2.remove()
    B, S = input_ids.shape
    hd = model.config.hidden_size // model.config.num_attention_heads
    q, k, _ = cap["qkv"].view(B, S, 3, model.config.num_attention_heads, hd).unbind(dim=-3)
    cos, sin = cap["pos"]
    q_rope, k_rope = apply_rotary_pos_emb(q.transpose(1, 2), k.transpose(1, 2), cos, sin, unsqueeze_dim=1)
    return q_rope, k_rope  # (B, heads, S, head_dim)


@pytest.mark.parametrize("truncate", [False, True])
def test_recorder_rope_qk_matches_model(truncate):
    model = tiny_modernbert()
    ref_q, ref_k = model_real_rope_qk(tiny_modernbert(), 4, torch.randint(3, 99, (2, 7)))

    recorder = create_recorder(model, 4, capture_rope=True)
    if truncate:
        recorder.truncate_layers()
    ids = torch.randint(3, 99, (2, 7))
    ref_q, ref_k = model_real_rope_qk(tiny_modernbert(), 4, ids)
    with torch.no_grad():
        model(input_ids=ids)
    recorder.pop()
    q_rope_bsh, k_rope_bsh = recorder.pop_rope_qk()

    B, S = ids.shape
    hd = model.config.hidden_size // model.config.num_attention_heads
    # recorder returns (B,S,H); reshape back to heads to compare with model form
    got_q = q_rope_bsh.view(B, S, model.config.num_attention_heads, hd).transpose(1, 2)
    got_k = k_rope_bsh.view(B, S, model.config.num_attention_heads, hd).transpose(1, 2)
    torch.testing.assert_close(got_q, ref_q, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(got_k, ref_k, rtol=1e-5, atol=1e-5)
    recorder.close()


def test_rope_qk_differs_from_content():
    """Sanity: the captured post-RoPE Q/K must actually differ from content
    Q/K (otherwise the whole experiment is a no-op)."""
    model = tiny_modernbert()
    recorder = create_recorder(model, 4, capture_rope=True)
    ids = torch.randint(3, 99, (2, 7))
    with torch.no_grad():
        model(input_ids=ids)
    q, k, _ = recorder.pop()
    q_rope, k_rope = recorder.pop_rope_qk()
    assert not torch.allclose(q, q_rope, atol=1e-3)
    assert not torch.allclose(k, k_rope, atol=1e-3)
    recorder.close()


def test_capture_rope_false_yields_no_rope():
    model = tiny_modernbert()
    recorder = create_recorder(model, 4)
    ids = torch.randint(3, 99, (2, 7))
    with torch.no_grad():
        model(input_ids=ids)
    recorder.pop()
    assert recorder.pop_rope_qk() is None
    recorder.close()


def test_bert_recorder_rejects_rope():
    from conftest import tiny_bert

    with pytest.raises(ValueError, match="RoPE"):
        create_recorder(tiny_bert(16, 2, 2), 1, capture_rope=True)


def _all_global_student(num_layers=2, hidden=16, heads=2):
    cfg = ModernBertConfig(
        vocab_size=99, hidden_size=hidden, num_hidden_layers=num_layers, num_attention_heads=heads,
        intermediate_size=hidden * 2, max_position_embeddings=64, local_attention=8,
        pad_token_id=0, bos_token_id=1, eos_token_id=2, cls_token_id=1, sep_token_id=2,
        reference_compile=False,
    )
    cfg.global_attn_every_n_layers = 1
    cfg.layer_types = ["full_attention"] * num_layers
    torch.manual_seed(5)
    return ModernBertModel(cfg)


@pytest.mark.parametrize("truncate", [False, True])
def test_hybrid_distiller_end_to_end(truncate):
    from train.distiller import MiniLMDistiller

    teacher = tiny_modernbert()
    student = _all_global_student()
    d = MiniLMDistiller(
        teacher=teacher, student=student, L=4, M=2,
        relations={(1, 1): 1.0, (2, 2): 1.0, (3, 3): 1.0, (4, 5): 1.0},
        A_r=4, teacher_dtype=torch.float32, truncate_teacher=truncate,
    )
    assert d._needs_rope
    ids = torch.randint(3, 99, (2, 8))
    mask = torch.ones(2, 8, dtype=torch.long)
    (loss,) = d(input_ids=ids, attention_mask=mask)
    assert torch.isfinite(loss) and loss.item() > 0
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in student.parameters())


def test_content_only_does_not_capture_rope():
    from train.distiller import MiniLMDistiller

    d = MiniLMDistiller(
        teacher=tiny_modernbert(), student=_all_global_student(), L=4, M=2,
        relations={(1, 1): 1.0, (2, 2): 1.0, (3, 3): 1.0}, A_r=4, teacher_dtype=torch.float32,
    )
    assert not d._needs_rope
    assert not d.teacher_recorder.capture_rope
