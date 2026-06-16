"""Break down where time goes in one distillation training step.

Times the teacher forward, student forward, loss, and backward separately
(with CUDA sync between phases), then runs torch.profiler for an op-level view.
"""
import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train.distiller import MiniLMDistiller  # noqa: E402
from train.losses import relation_kl, split_relation_heads  # noqa: E402


def build(args, device):
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    teacher = AutoModel.from_pretrained(args.teacher, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.teacher, trust_remote_code=True)
    cfg = AutoConfig.from_pretrained("bert-base-uncased")
    cfg.hidden_size = args.student_hidden_size
    cfg.num_hidden_layers = args.student_num_layers
    cfg.num_attention_heads = args.student_attention_heads
    cfg.intermediate_size = args.student_hidden_size * 4
    cfg.vocab_size = max(tokenizer.vocab_size, len(tokenizer))
    cfg.max_position_embeddings = max(cfg.max_position_embeddings, args.seq_len + 2)
    student = AutoModel.from_config(cfg)
    distiller = MiniLMDistiller(
        teacher=teacher, student=student, L=args.L, M=args.student_num_layers,
        relations={(1, 1): 1.0, (2, 2): 1.0, (3, 3): 1.0},
        A_r=args.num_relation_heads, head_chunk_size=args.head_chunk_size or None,
    ).to(device)
    student.train()
    return distiller, tokenizer.pad_token_id or 0


def batch(B, S, vocab, pad, device):
    ids = torch.randint(1, vocab, (B, S), device=device)
    lengths = torch.randint(3 * S // 4, S + 1, (B,), device=device)
    mask = (torch.arange(S, device=device)[None] < lengths[:, None]).long()
    ids[mask == 0] = pad
    return ids, mask


def phase_breakdown(distiller, ids, mask, A_r, reps):
    """Manually re-run the distiller's stages with timers between them."""
    teacher, student = distiller.teacher, distiller.student
    t_rec, s_rec = distiller.teacher_recorder, distiller.student_recorder
    timings = {k: 0.0 for k in ("teacher_fwd", "student_fwd", "loss", "backward")}

    def sync():
        torch.cuda.synchronize()

    for _ in range(reps):
        sync(); t0 = time.perf_counter()
        with torch.inference_mode():
            teacher(input_ids=ids, attention_mask=mask)
        qkv_t = tuple(x.clone() for x in t_rec.pop())
        sync(); t1 = time.perf_counter()

        student(input_ids=ids, attention_mask=mask)
        qkv_s = s_rec.pop()
        sync(); t2 = time.perf_counter()

        loss = relation_kl(
            [split_relation_heads(t, A_r) for t in qkv_t],
            [split_relation_heads(t, A_r) for t in qkv_s],
            distiller.relations, mask, head_chunk_size=distiller.head_chunk_size,
        )
        sync(); t3 = time.perf_counter()

        loss.backward()
        sync(); t4 = time.perf_counter()
        student.zero_grad(set_to_none=True)

        timings["teacher_fwd"] += t1 - t0
        timings["student_fwd"] += t2 - t1
        timings["loss"] += t3 - t2
        timings["backward"] += t4 - t3
    return {k: v / reps for k, v in timings.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--teacher", default="FacebookAI/xlm-roberta-large")
    p.add_argument("--student_hidden_size", type=int, default=768)
    p.add_argument("--student_num_layers", type=int, default=6)
    p.add_argument("--student_attention_heads", type=int, default=12)
    p.add_argument("--L", type=int, default=12)
    p.add_argument("--num_relation_heads", type=int, default=64)
    p.add_argument("--head_chunk_size", type=int, default=8)
    p.add_argument("--seq_len", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--reps", type=int, default=10)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--trace", help="Optional path to write a chrome trace.")
    args = p.parse_args()

    if not torch.cuda.is_available():
        sys.exit("CUDA required.")
    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"teacher={args.teacher} batch={args.batch_size} S={args.seq_len} "
          f"A_r={args.num_relation_heads} chunk={args.head_chunk_size}")

    distiller, pad = build(args, device)
    ids, mask = batch(args.batch_size, args.seq_len, distiller.student.config.vocab_size, pad, device)

    for _ in range(args.warmup):
        (loss,) = distiller(input_ids=ids, attention_mask=mask)
        loss.backward()
        distiller.student.zero_grad(set_to_none=True)

    timings = phase_breakdown(distiller, ids, mask, args.num_relation_heads, args.reps)
    total = sum(timings.values())
    print(f"\n--- per-step phase breakdown (mean of {args.reps}) ---")
    for name, t in sorted(timings.items(), key=lambda kv: -kv[1]):
        print(f"  {name:12s} {t * 1e3:8.2f} ms  {100 * t / total:5.1f}%")
    print(f"  {'TOTAL':12s} {total * 1e3:8.2f} ms")

    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        (loss,) = distiller(input_ids=ids, attention_mask=mask)
        loss.backward()
        distiller.student.zero_grad(set_to_none=True)
    print(f"\n--- top CUDA ops ---")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=12))
    if args.trace:
        prof.export_chrome_trace(args.trace)
        print(f"chrome trace -> {args.trace}")


if __name__ == "__main__":
    main()
