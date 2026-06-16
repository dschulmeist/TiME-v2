"""Trainer callbacks for distillation runs.

EfficiencyCallback records throughput/memory to efficiency.json; SaveStudentCallback
exports the bare student (config + weights + tokenizer) alongside each Trainer checkpoint.
"""
import json
import logging
import os
import time

import torch
from transformers import TrainerCallback, TrainerControl, TrainerState

logger = logging.getLogger(__name__)


class EfficiencyCallback(TrainerCallback):
    """Records throughput / memory efficiency of the run to efficiency.json.

    Captures wall-clock, steps/s, samples/s, real (non-pad) tokens/s, peak GPU
    memory, the student parameter count, and the active distillation terms, so
    runs can be compared on cost as well as quality.
    """

    def __init__(self, meta: dict):
        self._meta = meta
        self._t0 = None
        self._tokens = 0
        self._samples = 0

    def on_train_begin(self, args, state, control, **kwargs):
        self._t0 = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def on_step_end(self, args, state, control, **kwargs):
        # cheap running token/sample tally for a real-throughput number
        b = self._meta["batch"] * self._meta["grad_accum"] * max(1, self._meta["world_size"])
        self._samples += b
        self._tokens += b * self._meta["seq_len"]

    def on_train_end(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return
        elapsed = time.perf_counter() - (self._t0 or time.perf_counter())
        peak_gb = (torch.cuda.max_memory_allocated() / 1e9) if torch.cuda.is_available() else 0.0
        metrics = {
            **self._meta,
            "wall_clock_s": round(elapsed, 1),
            "steps": state.global_step,
            "steps_per_s": round(state.global_step / elapsed, 3) if elapsed else 0,
            "samples_per_s": round(self._samples / elapsed, 1) if elapsed else 0,
            "tokens_per_s": round(self._tokens / elapsed, 0) if elapsed else 0,
            "peak_vram_gb": round(peak_gb, 2),
        }
        path = os.path.join(args.output_dir, "efficiency.json")
        with open(path, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info("Efficiency metrics -> %s: %s", path, metrics)


class SaveStudentCallback(TrainerCallback):
    """On every Trainer checkpoint, also export the bare student (config +
    weights + tokenizer) under output_dir/student/checkpoint-<step>, since the
    Trainer only knows how to save the wrapping distiller."""

    def on_save(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        student_dir = os.path.join(args.output_dir, "student")
        ckpt_dir = os.path.join(student_dir, f"checkpoint-{state.global_step}")
        os.makedirs(ckpt_dir, exist_ok=True)
        student = kwargs["model"].student
        student.save_pretrained(ckpt_dir, safe_serialization=False)  # avoid safetensors (tied weights)
        student.config.save_pretrained(ckpt_dir)
        # transformers >=5 passes the tokenizer as "processing_class"
        tokenizer = kwargs.get("processing_class") or kwargs.get("tokenizer")
        if tokenizer is not None:
            tokenizer.save_pretrained(ckpt_dir)
        else:
            logger.warning("No tokenizer available in callback; student checkpoint has none.")

        # some transformers versions write an rng_state.pth that trips up resume;
        # drop the stray copies in the run root and the just-written checkpoint
        for rng in (
            os.path.join(args.output_dir, "rng_state.pth"),
            os.path.join(args.output_dir, f"checkpoint-{state.global_step}", "rng_state.pth"),
        ):
            try:
                os.remove(rng)
            except FileNotFoundError:
                pass
            except OSError as e:
                logger.warning("Could not remove RNG state %s: %s", rng, e)
        return control
