"""The single MiniLMv2 distiller used for all supported teacher architectures.

Architecture-specific Q/K/V extraction lives in :mod:`train.adapters`; the
loss lives in :mod:`train.losses`. This module only orchestrates the two
forward passes and applies the efficiency policies (frozen bf16 teacher,
teacher truncated at the distillation layer, head-chunked loss).
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch
from torch import nn

from .adapters import create_recorder, position_id_offset
from .losses import Relations, hidden_state_mse, logit_kd, relation_kl, split_relation_heads

logger = logging.getLogger(__name__)


class MiniLMDistiller(nn.Module):
    """MiniLMv2 multi-head self-attention relation distillation.

    Arguments
    ---------
    teacher, student : Hugging Face encoder models. The teacher is frozen.
    L, M             : 1-based layer to distill from (teacher) / to (student).
    relations        : {(i, j): weight}, 1=Q, 2=K, 3=V.
    A_r              : number of relation heads; must divide both hidden sizes.
    teacher_dtype    : dtype to cast the frozen teacher to. Default: bf16 on
                       CUDA, unchanged elsewhere.
    truncate_teacher : drop teacher layers above L (saves most teacher compute).
    head_chunk_size  : relation heads per loss chunk; bounds the peak memory of
                       the (B, A_r, S, S) relation matrices. None = no chunking.
    """

    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        L: int,
        M: int,
        relations: Relations,
        A_r: int,
        *,
        teacher_dtype: Optional[torch.dtype] = None,
        truncate_teacher: bool = True,
        head_chunk_size: Optional[int] = 8,
        compile_loss: bool = False,
        compile_student: bool = False,
        compile_teacher: bool = False,
        student_input_remap: Optional[torch.Tensor] = None,
        repr_weight: float = 0.0,
        logit_kd_weight: float = 0.0,
        logit_kd_temperature: float = 2.0,
        logit_kd_seq_chunk: Optional[int] = None,
    ) -> None:
        super().__init__()
        teacher_hidden, student_hidden = teacher.config.hidden_size, student.config.hidden_size
        for hidden, who in ((teacher_hidden, "teacher"), (student_hidden, "student")):
            if hidden % A_r != 0:
                raise ValueError(f"A_r={A_r} must divide the {who} hidden size {hidden}")

        self.teacher = teacher.eval()
        self.student = student
        self.relations = dict(relations)
        self.A_r = A_r
        self.head_chunk_size = head_chunk_size
        self.repr_weight = repr_weight
        self.logit_kd_weight = logit_kd_weight
        self.logit_kd_temperature = logit_kd_temperature
        self.logit_kd_seq_chunk = logit_kd_seq_chunk
        self._needs_hidden = repr_weight > 0
        self._needs_logit = logit_kd_weight > 0
        # relation indices: 1=Q 2=K 3=V (content, pre-RoPE) and 4=Q 5=K
        # post-RoPE - the position-aware Q-K term needs the rotated vectors
        self._needs_rope = any(i >= 4 for rel in self.relations for i in rel)
        self._relation_loss = torch.compile(relation_kl) if compile_loss else relation_kl

        # hidden-state term: project the student's width to the teacher's
        # (the only auxiliary trainable params; dropped at export)
        self.hidden_projection = (
            nn.Linear(student_hidden, teacher_hidden) if self._needs_hidden else None
        )
        if self._needs_logit:
            if student_input_remap is not None:
                raise ValueError(
                    "logit-KD needs teacher and student to share a vocabulary, which "
                    "is incompatible with --prune_student_vocab (the student emits "
                    "different ids). Disable one of them."
                )
            if teacher.config.vocab_size != student.config.vocab_size:
                raise ValueError(
                    f"logit-KD needs a shared vocabulary; teacher vocab "
                    f"{teacher.config.vocab_size} != student {student.config.vocab_size}."
                )
        # maps full-vocab ids to the student's pruned embedding rows (see
        # train/vocab_pruning.py); the teacher always sees the original ids
        self.register_buffer("student_input_remap", student_input_remap, persistent=False)
        if compile_student:
            # in-place compile keeps state_dict keys and the Q/K/V hooks intact
            # (hook side effects cause a graph break at the hooked layer, not
            # wrong results)
            self.student.compile()

        for p in self.teacher.parameters():
            p.requires_grad_(False)
        if teacher_dtype is None and torch.cuda.is_available():
            teacher_dtype = torch.bfloat16
        if teacher_dtype is not None:
            self.teacher.to(teacher_dtype)

        self.teacher_recorder = create_recorder(
            self.teacher, L, capture_rope=self._needs_rope, capture_hidden=self._needs_hidden)
        self.student_recorder = create_recorder(
            self.student, M, capture_rope=self._needs_rope, capture_hidden=self._needs_hidden)
        # logit-KD reads the teacher's output logits, so the full teacher
        # (all layers + LM head) must run - truncation is mutually exclusive.
        if truncate_teacher and self._needs_logit:
            logger.info("logit-KD active: keeping the full teacher (truncation disabled).")
            truncate_teacher = False
        if truncate_teacher:
            self.teacher_recorder.truncate_layers()
            logger.info("Teacher truncated to %d layers for distillation.", L)
        if compile_teacher:
            self.teacher.compile()

        pad_token_id = getattr(self.teacher.config, "pad_token_id", None)
        self._pad_token_id = pad_token_id
        logger.info(
            "MiniLMDistiller: teacher layer %d -> student layer %d, relations=%s, A_r=%d, "
            "teacher dtype=%s, head_chunk_size=%s",
            L, M, self.relations, A_r, next(self.teacher.parameters()).dtype, head_chunk_size,
        )

    def state_dict(self, *args, **kwargs):
        """Checkpoint only the trainable parts (student + projection), cloned.

        The Trainer saves a non-PreTrainedModel via safetensors, which rejects
        shared storage - and an MLM student/teacher tie their decoder to their
        embeddings. We drop the frozen teacher (rebuilt from its pretrained
        weights on resume) and clone the rest to break the tying.
        """
        sd = super().state_dict(*args, **kwargs)
        return {k: v.detach().clone() for k, v in sd.items() if not k.startswith("teacher.")}

    def load_state_dict(self, state_dict, strict: bool = True, **kwargs):
        # Our checkpoints omit the frozen teacher (rebuilt from pretrained on
        # resume), so a strict load would always fail on the teacher keys. We
        # load non-strictly, but ALWAYS verify every trainable (student /
        # projection) key is present - a genuinely missing trainable weight
        # must never pass silently, even on Trainer resume (which calls us with
        # strict=False). Only teacher-key absence is tolerated.
        result = super().load_state_dict(state_dict, strict=False, **kwargs)
        missing_trainable = [k for k in result.missing_keys if not k.startswith("teacher.")]
        if missing_trainable or result.unexpected_keys:
            raise RuntimeError(
                f"distiller load_state_dict: missing trainable keys "
                f"{missing_trainable}, unexpected keys {list(result.unexpected_keys)}"
            )
        return result

    def train(self, mode: bool = True) -> "MiniLMDistiller":
        """Keep the frozen teacher in eval mode (no dropout) regardless of the
        Trainer toggling the distiller's mode."""
        super().train(mode)
        self.teacher.eval()
        return self

    def gradient_checkpointing_enable(self, **kwargs) -> None:
        """Delegate to the student so `TrainingArguments(gradient_checkpointing=True)`
        works on the distiller. The frozen teacher stores no activations for
        backward, so checkpointing it would only cost compute."""
        self.student.gradient_checkpointing_enable(**kwargs)

    def gradient_checkpointing_disable(self) -> None:
        self.student.gradient_checkpointing_disable()

    def _model_inputs(self, model, recorder, input_ids, attention_mask, token_type_ids, segment_ids, position_ids):
        inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None and recorder.accepts_token_type_ids:
            inputs["token_type_ids"] = token_type_ids
        if segment_ids is not None:
            inputs["attention_mask"] = recorder.build_packed_attention(attention_mask, segment_ids)
            inputs["position_ids"] = position_ids + position_id_offset(model.config)
        return inputs

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        segment_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        **_: object,
    ) -> Tuple[torch.Tensor]:
        if attention_mask is None:
            if self._pad_token_id is None:
                raise ValueError(
                    "attention_mask not provided and the teacher config defines no "
                    "pad_token_id - cannot infer padding. Pass an attention_mask."
                )
            attention_mask = (input_ids != self._pad_token_id).long()
        if segment_ids is not None:
            if position_ids is None:
                raise ValueError("packed inputs require position_ids alongside segment_ids")
            for recorder, who in ((self.teacher_recorder, "teacher"), (self.student_recorder, "student")):
                if not recorder.supports_packing:
                    raise ValueError(
                        f"The {who} architecture does not support packed inputs "
                        f"(block-diagonal masks are not equivalence-preserving for it). "
                        f"Train without --pack."
                    )

        with torch.inference_mode():
            t_out = self.teacher(**self._model_inputs(
                self.teacher, self.teacher_recorder,
                input_ids, attention_mask, token_type_ids, segment_ids, position_ids,
            ))
        # clone: inference-mode tensors cannot participate in the autograd graph
        qkv_teacher = [t.clone() for t in self._collect(self.teacher_recorder)]
        hidden_teacher = self._clone(self.teacher_recorder.pop_hidden())
        logits_teacher = self._clone(getattr(t_out, "logits", None)) if self._needs_logit else None

        student_ids = input_ids if self.student_input_remap is None else self.student_input_remap[input_ids]
        s_out = self.student(**self._model_inputs(
            self.student, self.student_recorder,
            student_ids, attention_mask, token_type_ids, segment_ids, position_ids,
        ))
        qkv_student = self._collect(self.student_recorder)

        loss = self._relation_loss(
            [split_relation_heads(t, self.A_r) for t in qkv_teacher],
            [split_relation_heads(t, self.A_r) for t in qkv_student],
            self.relations,
            attention_mask,
            head_chunk_size=self.head_chunk_size,
            segment_ids=segment_ids,
        )
        if self._needs_hidden:
            loss = loss + self.repr_weight * hidden_state_mse(
                hidden_teacher, self.student_recorder.pop_hidden(),
                attention_mask, self.hidden_projection,
            )
        if self._needs_logit:
            if not hasattr(s_out, "logits"):
                raise RuntimeError(
                    "logit-KD requires models with an LM head (load with "
                    "AutoModelForMaskedLM); the student returned no .logits."
                )
            loss = loss + self.logit_kd_weight * logit_kd(
                logits_teacher, s_out.logits, attention_mask, self.logit_kd_temperature,
                seq_chunk_size=self.logit_kd_seq_chunk,
            )
        return (loss,)

    @staticmethod
    def _clone(t):
        return t.clone() if t is not None else None

    def _collect(self, recorder):
        """(q, k, v) - extended with post-RoPE (q, k) at indices 4,5 (1-based)
        when the relation set uses position-aware Q-K."""
        qkv = list(recorder.pop())
        if self._needs_rope:
            q_rope, k_rope = recorder.pop_rope_qk()
            qkv += [q_rope, k_rope]
        return qkv
