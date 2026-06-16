import pytest
import torch
from torch import nn

from train.adapters import BertLikeRecorder, LtgRecorder, create_recorder


class FakeLtgConfig:
    model_type = "ltgbert"

    def __init__(self, hidden_size, num_hidden_layers):
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers


class FakeLtgAttention(nn.Module):
    """Mimics LTG-BERT attention: fused in_proj_qk, separate in_proj_v,
    (S, B, H) activations, and an extra 2D in_proj_qk call for the
    relative-embedding table that recorders must ignore."""

    def __init__(self, hidden_size):
        super().__init__()
        self.in_proj_qk = nn.Linear(hidden_size, 2 * hidden_size)
        self.in_proj_v = nn.Linear(hidden_size, hidden_size)
        self.relative_embedding = nn.Parameter(torch.randn(7, hidden_size))

    def forward(self, x_sbh):
        # the decoy 2D call (relative position projections in real LTG-BERT)
        self.in_proj_qk(self.relative_embedding)
        qk = self.in_proj_qk(x_sbh)
        v = self.in_proj_v(x_sbh)
        q, _ = qk.chunk(2, dim=-1)
        return q + v  # arbitrary mixing; only the hooks matter for the test


class FakeLtgLayer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = FakeLtgAttention(hidden_size)

    def forward(self, x_sbh):
        return self.attention(x_sbh)


class FakeLtgModel(nn.Module):
    def __init__(self, hidden_size=16, num_layers=2, vocab=50):
        super().__init__()
        self.config = FakeLtgConfig(hidden_size, num_layers)
        self.embedding = nn.Embedding(vocab, hidden_size)
        self.transformer = nn.Module()
        self.transformer.layers = nn.ModuleList(
            FakeLtgLayer(hidden_size) for _ in range(num_layers)
        )

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        x = self.embedding(input_ids).transpose(0, 1)  # (B,S,H) -> (S,B,H)
        for layer in self.transformer.layers:
            x = layer(x)
        return x.transpose(0, 1)


def test_ltg_recorder_extracts_and_transposes_qkv():
    torch.manual_seed(0)
    model = FakeLtgModel()
    recorder = create_recorder(model, layer=2)
    assert isinstance(recorder, LtgRecorder)

    input_ids = torch.randint(0, 50, (3, 5))
    with torch.no_grad():
        model(input_ids)
    q, k, v = recorder.pop()

    # manual recompute of the layer-2 input and projections, in (B, S, H)
    with torch.no_grad():
        x = model.embedding(input_ids).transpose(0, 1)
        x = model.transformer.layers[0](x)
        attn = model.transformer.layers[1].attention
        qk = attn.in_proj_qk(x)
        q_ref, k_ref = (t.transpose(0, 1) for t in qk.chunk(2, dim=-1))
        v_ref = attn.in_proj_v(x).transpose(0, 1)

    assert q.shape == (3, 5, 16)
    torch.testing.assert_close(q, q_ref)
    torch.testing.assert_close(k, k_ref)
    torch.testing.assert_close(v, v_ref)
    recorder.close()


def test_ltg_recorder_truncates_layers():
    model = FakeLtgModel()
    recorder = create_recorder(model, layer=1)
    recorder.truncate_layers()
    assert len(model.transformer.layers) == 1
    with torch.no_grad():
        model(torch.randint(0, 50, (2, 4)))
    q, k, v = recorder.pop()
    assert q.shape == (2, 4, 16)
    recorder.close()


def test_hooked_qkv_equals_manual_recompute(teacher, batch):
    """The recorded Q/K/V must equal applying the layer's own projections to the
    layer input (catches dropped-bias and wrong-layer bugs)."""
    L = 2
    recorder = BertLikeRecorder(teacher, L)
    with torch.no_grad():
        outputs = teacher(**batch, output_hidden_states=True)
    q, k, v = recorder.pop()

    layer_input = outputs.hidden_states[L - 1]
    attn = teacher.encoder.layer[L - 1].attention.self
    with torch.no_grad():
        torch.testing.assert_close(q, attn.query(layer_input))
        torch.testing.assert_close(k, attn.key(layer_input))
        torch.testing.assert_close(v, attn.value(layer_input))
    recorder.close()


def test_registry_dispatch_and_unknown_architecture(teacher):
    recorder = create_recorder(teacher, 1)
    assert isinstance(recorder, BertLikeRecorder)
    recorder.close()

    teacher.config.model_type = "some-unknown-arch"
    with pytest.raises(ValueError, match="some-unknown-arch"):
        create_recorder(teacher, 1)


def test_layer_out_of_range(teacher):
    with pytest.raises(ValueError, match="out of range"):
        BertLikeRecorder(teacher, 99)


def test_pop_clears_cache_and_errors_when_empty(teacher, batch):
    recorder = BertLikeRecorder(teacher, 1)
    with torch.no_grad():
        teacher(**batch)
    recorder.pop()
    with pytest.raises(RuntimeError, match="incomplete"):
        recorder.pop()
    recorder.close()


def test_truncate_layers_preserves_recorded_qkv(teacher, batch):
    """Truncating the teacher above layer L must not change the recorded Q/K/V."""
    L = 2
    recorder = BertLikeRecorder(teacher, L)
    with torch.no_grad():
        teacher(**batch)
    full_qkv = recorder.pop()

    recorder.truncate_layers()
    assert len(teacher.encoder.layer) == L
    assert teacher.config.num_hidden_layers == L
    with torch.no_grad():
        teacher(**batch)
    truncated_qkv = recorder.pop()

    for full, truncated in zip(full_qkv, truncated_qkv):
        torch.testing.assert_close(truncated, full)
    recorder.close()


def test_close_removes_hooks(teacher, batch):
    recorder = BertLikeRecorder(teacher, 1)
    recorder.close()
    with torch.no_grad():
        teacher(**batch)
    assert recorder._cache == {}
