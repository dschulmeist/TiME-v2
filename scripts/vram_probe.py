#!/usr/bin/env python
"""Measure peak VRAM and step time of the distillation training step.

Builds the same distiller the Trainer would, runs a few synthetic optimizer
steps per micro-batch size, and reports peak memory, step time, and the
gradient-accumulation factor needed for the target global batch:

    uv run python scripts/vram_probe.py --teacher FacebookAI/xlm-roberta-large \
        --L 12 --num_relation_heads 64 --batch_sizes 8,16,32,64
"""
import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train.distiller import MiniLMDistiller  # noqa: E402


def build_models(args):
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    teacher = AutoModel.from_pretrained(args.teacher, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.teacher, trust_remote_code=True)

    student_config = AutoConfig.from_pretrained("bert-base-uncased")
    student_config.hidden_size = args.student_hidden_size
    student_config.num_hidden_layers = args.student_num_layers
    student_config.num_attention_heads = args.student_attention_heads
    student_config.intermediate_size = args.student_hidden_size * 4
    student_config.vocab_size = max(tokenizer.vocab_size, len(tokenizer))
    student_config.max_position_embeddings = max(
        student_config.max_position_embeddings, args.seq_len + 2
    )
    student = AutoModel.from_config(student_config)
    return teacher, student, tokenizer


def synthetic_batch(batch_size, seq_len, vocab_size, pad_token_id, device):
    """Random ids with ragged lengths (3/4 to full) to exercise the masking."""
    input_ids = torch.randint(1, vocab_size, (batch_size, seq_len), device=device)
    lengths = torch.randint(3 * seq_len // 4, seq_len + 1, (batch_size,), device=device)
    positions = torch.arange(seq_len, device=device)[None, :]
    attention_mask = (positions < lengths[:, None]).long()
    input_ids[attention_mask == 0] = pad_token_id
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def probe(distiller, optimizer, batch, steps):
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    times = []
    for _ in range(steps):
        start = time.perf_counter()
        (loss,) = distiller(**batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    return {
        "loss": loss.item(),
        "step_time_s": min(times),
        "peak_alloc_gb": torch.cuda.max_memory_allocated() / 1e9,
        "peak_reserved_gb": torch.cuda.max_memory_reserved() / 1e9,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", default="FacebookAI/xlm-roberta-large")
    parser.add_argument("--student_hidden_size", type=int, default=768)
    parser.add_argument("--student_num_layers", type=int, default=6)
    parser.add_argument("--student_attention_heads", type=int, default=12)
    parser.add_argument("--L", type=int, default=12)
    parser.add_argument("--num_relation_heads", type=int, default=64)
    parser.add_argument("--head_chunk_size", type=int, default=8)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--batch_sizes", default="8,16,32,64")
    parser.add_argument("--steps", type=int, default=4, help="Steps per batch size (best is reported).")
    parser.add_argument("--target_global_batch", type=int, default=256)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        sys.exit("CUDA is required - run this on the GPU box.")
    device = torch.device("cuda")
    total_gb = torch.cuda.get_device_properties(device).total_memory / 1e9
    print(f"GPU: {torch.cuda.get_device_name(device)} ({total_gb:.1f} GB)")

    teacher, student, tokenizer = build_models(args)
    distiller = MiniLMDistiller(
        teacher=teacher,
        student=student,
        L=args.L,
        M=args.student_num_layers,
        relations={(1, 1): 1.0, (2, 2): 1.0, (3, 3): 1.0},
        A_r=args.num_relation_heads,
        head_chunk_size=args.head_chunk_size or None,
    ).to(device)
    if args.gradient_checkpointing:
        distiller.gradient_checkpointing_enable()
    student.train()
    try:
        from bitsandbytes.optim import AdamW8bit
        optimizer = AdamW8bit(student.parameters(), lr=6e-4)
        print("Optimizer: 8-bit AdamW (bitsandbytes)")
    except ImportError:
        optimizer = torch.optim.AdamW(student.parameters(), lr=6e-4)
        print("Optimizer: fp32 AdamW (bitsandbytes not installed)")

    print(
        f"Teacher params (truncated, bf16): {sum(p.numel() for p in teacher.parameters()) / 1e6:.0f}M | "
        f"Student params: {sum(p.numel() for p in student.parameters()) / 1e6:.0f}M"
    )
    print(f"{'batch':>6} {'accum':>6} {'loss':>8} {'s/step':>8} {'alloc GB':>9} {'reserved GB':>12}")

    for batch_size in (int(b) for b in args.batch_sizes.split(",")):
        accum = -(-args.target_global_batch // batch_size)
        batch = synthetic_batch(
            batch_size, args.seq_len, student.config.vocab_size,
            tokenizer.pad_token_id or 0, device,
        )
        try:
            stats = probe(distiller, optimizer, batch, args.steps)
        except torch.cuda.OutOfMemoryError:
            print(f"{batch_size:>6} {accum:>6} {'OOM':>8}")
            break
        finally:
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
        print(
            f"{batch_size:>6} {accum:>6} {stats['loss']:>8.4f} {stats['step_time_s']:>8.3f} "
            f"{stats['peak_alloc_gb']:>9.2f} {stats['peak_reserved_gb']:>12.2f}"
        )


if __name__ == "__main__":
    main()
