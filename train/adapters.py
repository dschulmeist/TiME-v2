"""Architecture-specific Q/K/V extraction for MiniLMv2 distillation.

Each recorder wraps a Hugging Face model, hooks the Q/K/V projections of one
encoder layer, and exposes the captured (B, S, H) tensors via :meth:`pop`.
New teacher architectures are added by subclassing :class:`QKVRecorder` and
registering the model_type - never by string-sniffing architecture names.
"""
from __future__ import annotations

from typing import Dict, List, Tuple, Type

import torch
from torch import nn

_REGISTRY: Dict[str, Type["QKVRecorder"]] = {}


def register(*model_types: str):
    """Class decorator: register a recorder for one or more config.model_type values."""

    def wrap(cls: Type["QKVRecorder"]) -> Type["QKVRecorder"]:
        for mt in model_types:
            _REGISTRY[mt] = cls
        return cls

    return wrap


def position_id_offset(config) -> int:
    """RoBERTa-family models reserve position ids 0..pad for special use;
    real tokens start at pad_token_id + 1. BERT-style models start at 0."""
    if config.model_type in ("roberta", "xlm-roberta", "camembert"):
        pad = config.pad_token_id if config.pad_token_id is not None else 1
        return pad + 1
    return 0


def create_recorder(model: nn.Module, layer: int, capture_rope: bool = False,
                    capture_hidden: bool = False) -> "QKVRecorder":
    """Instantiate the recorder registered for `model.config.model_type`.

    Raises a hard error for unknown architectures instead of silently assuming
    a BERT layout (which crashes later, or worse, hooks the wrong tensors).
    `capture_rope` requests post-RoPE Q/K (only RoPE recorders honor it);
    `capture_hidden` also records the hooked layer's output hidden state.
    """
    model_type = getattr(model.config, "model_type", None)
    cls = _REGISTRY.get(model_type)
    if cls is None:
        raise ValueError(
            f"No Q/K/V recorder registered for model_type '{model_type}'. "
            f"Supported: {sorted(_REGISTRY)}. Add a QKVRecorder subclass in "
            f"train/adapters.py to support this architecture."
        )
    if capture_rope and not cls.supports_rope_qk:
        raise ValueError(f"{cls.__name__} (model_type '{model_type}') has no RoPE Q/K to capture")
    kwargs = {"capture_hidden": capture_hidden}
    if capture_rope:
        kwargs["capture_rope"] = True
    return cls(model, layer, **kwargs)


class QKVRecorder:
    """Hooks one encoder layer of `model` and records its Q/K/V projections.

    `layer` is 1-based, matching the paper's and the CLI's --L convention.
    """

    #: whether the wrapped model's forward accepts token_type_ids
    accepts_token_type_ids: bool = True
    #: whether a block-diagonal 4D mask reproduces isolated per-document forwards
    supports_packing: bool = True

    def __init__(self, model: nn.Module, layer: int, capture_hidden: bool = False) -> None:
        self.model = model
        self.layer = layer
        self.capture_hidden = capture_hidden
        self._cache: Dict[str, torch.Tensor] = {}
        self._handles: List[torch.utils.hooks.RemovableHandle] = []
        num_layers = len(self._layers())
        if not 1 <= layer <= num_layers:
            raise ValueError(
                f"Layer {layer} out of range for {type(model).__name__} with {num_layers} layers"
            )
        self._attach()
        if capture_hidden:
            self._handles.append(
                self._layers()[self.layer - 1].register_forward_hook(self._record_hidden)
            )

    def _record_hidden(self, module, inputs, output) -> None:
        # encoder layers return either the hidden-state tensor (ModernBERT) or a
        # tuple whose first element is it (BERT family)
        self._cache["hidden"] = output[0] if isinstance(output, tuple) else output

    # -- subclass interface --------------------------------------------------

    def _layers(self) -> nn.ModuleList:
        """Return the model's encoder layer list."""
        raise NotImplementedError

    def _attach(self) -> None:
        """Register forward hooks that fill self._cache with 'q', 'k', 'v'."""
        raise NotImplementedError

    # -- public API ----------------------------------------------------------

    def build_packed_attention(self, attention_mask: torch.Tensor, segment_ids: torch.Tensor):
        """Attention input restricting packed rows to within-segment pairs.

        Returns a 4D additive mask; architectures with heterogeneous layer
        types override this.
        """
        real = attention_mask.bool()
        allowed = (segment_ids[:, :, None] == segment_ids[:, None, :]) & real[:, :, None] & real[:, None, :]
        dtype = next(self.model.parameters()).dtype
        mask = torch.zeros(allowed.shape, dtype=dtype, device=allowed.device)
        return mask.masked_fill_(~allowed, torch.finfo(dtype).min)[:, None]

    #: whether this recorder can also supply post-RoPE Q/K (position-aware
    #: relations); only meaningful for RoPE architectures
    supports_rope_qk: bool = False

    def pop(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the recorded (q, k, v), each (B, S, H), and clear the cache."""
        missing = [k for k in ("q", "k", "v") if k not in self._cache]
        if missing:
            raise RuntimeError(
                f"Q/K/V cache incomplete after forward pass (missing {missing}). "
                f"The hooked modules did not run - check layer index and model structure."
            )
        q, k, v = self._cache["q"], self._cache["k"], self._cache["v"]
        rope = (self._cache.get("q_rope"), self._cache.get("k_rope"))
        self._rope_qk = rope if rope[0] is not None else None
        self._hidden = self._cache.get("hidden")
        self._cache.clear()
        return q, k, v

    def pop_rope_qk(self):
        """Post-RoPE (q, k) from the most recent pop(), each (B, S, H), or None.

        Set by pop() so a single forward yields both content and position-aware
        Q/K. None unless the recorder captured RoPE this pass.
        """
        return getattr(self, "_rope_qk", None)

    def pop_hidden(self):
        """Layer-output hidden state (B, S, H) from the most recent pop(), or
        None unless capture_hidden was requested."""
        return getattr(self, "_hidden", None)

    def truncate_layers(self) -> None:
        """Drop all encoder layers above the hooked one, and reduce the hooked
        layer itself to its Q/K/V projection where the architecture allows.

        Everything past the Wqkv/Q-K-V projections of layer L contributes
        nothing to distillation; for a frozen teacher this removes the
        majority of its forward compute.
        """
        layers = self._layers()
        if len(layers) > self.layer:
            del layers[self.layer :]
            self.model.config.num_hidden_layers = self.layer
        if self.capture_hidden:
            return  # the full layer-L output is needed; can't reduce it to a stub
        stub = self._distill_layer_stub(layers[self.layer - 1])
        if stub is not None:
            layers[self.layer - 1] = stub
            self._on_stub_installed()

    def _distill_layer_stub(self, layer: nn.Module) -> nn.Module | None:
        """A replacement for the hooked layer that computes only what the
        hooks observe. None = keep the full layer (default)."""
        return None

    def _on_stub_installed(self) -> None:
        """Called after the hooked layer is swapped for a stub, so subclasses
        can re-attach hooks that were on the now-detached layer."""

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def __del__(self) -> None:
        self.close()


@register("bert", "roberta", "xlm-roberta", "camembert", "electra")
class BertLikeRecorder(QKVRecorder):
    """BERT-family models with separate query/key/value linears at
    `encoder.layer[i].attention.self`. Hooks fire under any attention
    implementation (eager/sdpa/flash) since the projections are modules.
    """

    def _layers(self) -> nn.ModuleList:
        encoder = self.model.encoder if hasattr(self.model, "encoder") else self.model.base_model.encoder
        return encoder.layer

    def _attach(self) -> None:
        attn = self._layers()[self.layer - 1].attention.self
        for name, module in (("q", attn.query), ("k", attn.key), ("v", attn.value)):
            self._handles.append(
                module.register_forward_hook(
                    lambda mod, inp, out, name=name: self._cache.__setitem__(name, out)
                )
            )


class _ModernBertWqkvOnly(nn.Module):
    """Replaces the distillation layer of a truncated ModernBERT teacher.

    Only the Wqkv projection (which the recorder hooks) is computed; the
    layer's attention and MLP outputs are never consumed, so the input is
    passed through unchanged. Reuses the original submodules, keeping
    registered hooks attached.
    """

    def __init__(self, layer: nn.Module) -> None:
        super().__init__()
        self.attn_norm = layer.attn_norm
        self.Wqkv = layer.attn.Wqkv
        # the encoder reads this to select the per-layer-type attention mask
        self.attention_type = layer.attention_type

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        self.Wqkv(self.attn_norm(hidden_states))
        return hidden_states


@register("modernbert")
class ModernBertRecorder(QKVRecorder):
    """ModernBERT: fused Wqkv projection at `layers[i].attn`, no token-type
    embeddings, RoPE positions. Q/K/V are tapped pre-RoPE, so the recorded
    relations are content-only - consistent for Q-Q/K-K/V-V (V is never
    rotated), and it avoids ModernBERT's split local/global rope_theta.

    The distillation layer must use full (global) attention: the loss computes
    dense relations the teacher never formed inside a sliding window.
    Load the teacher with attn_implementation="sdpa" or "eager" - FA2 unpads
    the batch, breaking the (B, S, H) hook contract.
    """

    accepts_token_type_ids = False
    supports_rope_qk = True

    def __init__(self, model: nn.Module, layer: int, capture_rope: bool = False,
                 capture_hidden: bool = False) -> None:
        self.capture_rope = capture_rope
        super().__init__(model, layer, capture_hidden=capture_hidden)
        layer_types = getattr(model.config, "layer_types", None)
        if layer_types is not None and layer_types[layer - 1] != "full_attention":
            full = [i + 1 for i, t in enumerate(layer_types) if t == "full_attention"]
            raise ValueError(
                f"Layer {layer} uses sliding-window attention; distill from a "
                f"global-attention layer instead. Valid choices: {full}"
            )

    def _layers(self) -> nn.ModuleList:
        return self.model.layers if hasattr(self.model, "layers") else self.model.model.layers

    def _attach(self) -> None:
        attn = self._layers()[self.layer - 1].attn
        self._handles.append(attn.Wqkv.register_forward_hook(self._record))
        if self.capture_rope:
            # the layer (real or surgical stub) receives position_embeddings
            # (cos, sin); capture them before Wqkv fires so _record can rotate
            self._handles.append(
                self._layers()[self.layer - 1].register_forward_pre_hook(
                    self._capture_pos, with_kwargs=True
                )
            )

    def _capture_pos(self, module, args, kwargs) -> None:
        pos = kwargs.get("position_embeddings")
        if pos is None:
            raise RuntimeError(
                "capture_rope=True but the layer received no position_embeddings; "
                "the model must be run normally (not in a path that drops RoPE)."
            )
        self._cache["_pos"] = pos

    def build_packed_attention(self, attention_mask: torch.Tensor, segment_ids: torch.Tensor):
        """Per-layer-type mask dict with the same-segment constraint injected
        into transformers' own mask builders, so sliding-window semantics stay
        exactly the model's own."""
        from transformers.masking_utils import (
            create_bidirectional_mask,
            create_bidirectional_sliding_window_mask,
        )

        def same_segment(batch_idx, head_idx, q_idx, kv_idx):
            return segment_ids[batch_idx, q_idx] == segment_ids[batch_idx, kv_idx]

        B, S = attention_mask.shape
        param = next(self.model.parameters())
        mask_kwargs = {
            "config": self.model.config,
            "inputs_embeds": torch.empty(B, S, 0, dtype=param.dtype, device=param.device),
            "attention_mask": attention_mask,
            "and_mask_function": same_segment,
        }
        return {
            "full_attention": create_bidirectional_mask(**mask_kwargs),
            "sliding_attention": create_bidirectional_sliding_window_mask(**mask_kwargs),
        }

    def _distill_layer_stub(self, layer: nn.Module) -> nn.Module:
        return _ModernBertWqkvOnly(layer)

    def truncate_layers(self) -> None:
        super().truncate_layers()
        if self.capture_rope:
            # the base swapped layer L for the Wqkv-only stub; the Wqkv hook
            # survived (stub reuses the module) but the position pre-hook was on
            # the now-detached layer, so re-attach it to the stub (which the
            # encoder still passes position_embeddings to, keyed by attention_type)
            self._handles.append(
                self._layers()[self.layer - 1].register_forward_pre_hook(
                    self._capture_pos, with_kwargs=True
                )
            )

    def _record(self, module: nn.Module, inputs: tuple, output: torch.Tensor) -> None:
        if output.dim() != 3:
            raise RuntimeError(
                f"Wqkv produced a {output.dim()}D tensor - the batch was unpadded "
                f"(flash-attention path or a transformers version that re-introduced "
                f"full-model unpadding). Load the model with attn_implementation='sdpa' "
                f"and keep the pinned transformers version."
            )
        B, S, _ = output.shape
        heads = self.model.config.num_attention_heads
        head_dim = self.model.config.hidden_size // heads
        q, k, v = output.view(B, S, 3, heads, head_dim).unbind(dim=-3)
        self._cache["q"] = q.reshape(B, S, heads * head_dim)
        self._cache["k"] = k.reshape(B, S, heads * head_dim)
        self._cache["v"] = v.reshape(B, S, heads * head_dim)
        if self.capture_rope:
            from transformers.models.modernbert.modeling_modernbert import apply_rotary_pos_emb
            cos, sin = self._cache["_pos"]
            # RoPE acts in attention-head space; rotate there, then flatten
            # back to (B, S, H) for the relation-head split downstream
            qh = q.transpose(1, 2)  # (B, heads, S, head_dim)
            kh = k.transpose(1, 2)
            q_rope, k_rope = apply_rotary_pos_emb(qh, kh, cos, sin, unsqueeze_dim=1)
            self._cache["q_rope"] = q_rope.transpose(1, 2).reshape(B, S, heads * head_dim)
            self._cache["k_rope"] = k_rope.transpose(1, 2).reshape(B, S, heads * head_dim)
            del self._cache["_pos"]


@register("ltgbert", "ltg-bert")
class LtgRecorder(QKVRecorder):
    """HPLT / LTG-BERT models: fused in_proj_qk plus separate in_proj_v, with
    (S, B, H) layout that is transposed back to (B, S, H) on capture.
    """

    # LTG's custom modeling code has not been validated with 4D block masks
    supports_packing = False

    def _layers(self) -> nn.ModuleList:
        return self.model.transformer.layers

    def _attach(self) -> None:
        attn = self._layers()[self.layer - 1].attention
        self._handles.append(
            attn.in_proj_qk.register_forward_hook(self._record_qk)
        )
        self._handles.append(
            attn.in_proj_v.register_forward_hook(self._record_v)
        )

    def _record_qk(self, module: nn.Module, inputs: tuple, output: torch.Tensor) -> None:
        # LTG attention also calls in_proj_qk on the relative-embedding table,
        # whose input is 2D - only the (S, B, H) token activations are wanted.
        if inputs[0].dim() != 3:
            return
        q, k = output.chunk(2, dim=-1)
        self._cache["q"] = q.transpose(0, 1)
        self._cache["k"] = k.transpose(0, 1)

    def _record_v(self, module: nn.Module, inputs: tuple, output: torch.Tensor) -> None:
        if inputs[0].dim() != 3:
            return
        self._cache["v"] = output.transpose(0, 1)
