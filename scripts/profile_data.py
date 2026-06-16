"""Profile the input pipeline: streaming load + tokenize + collate throughput.

Measures pure data throughput (no model) and, optionally, how much the
dataloader stalls a real training step - i.e. whether data or compute is the
bottleneck.
"""
import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class _Cfg:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def build_loader(args):
    from torch.utils.data import DataLoader
    from transformers import DataCollatorWithPadding

    from train.data_pipeline import get_tokenized_datasets

    data_args = _Cfg(
        dataset_name=args.dataset, dataset_config_name=args.config,
        streaming=args.streaming, max_seq_len=args.seq_len,
        is_local_arrow_config=False, local_arrow_files_config=None,
        stream_local_files=False, text_column_name="text",
        shuffle_buffer_size=args.shuffle_buffer, stream_take_size=0,
        stream_take_size_eval=0, map_batch_size=1000, do_eval=False,
        preprocessing_num_workers=None, overwrite_cache=False,
    )
    train_ds, _, tokenizer = get_tokenized_datasets(
        data_args, _Cfg(tokenizer_name_or_path=args.tokenizer), _Cfg(seed=21)
    )
    train_ds = train_ds.with_format("torch")
    loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=DataCollatorWithPadding(tokenizer, padding="longest"),
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=args.prefetch if args.num_workers > 0 else None,
        drop_last=True,
    )
    return loader


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="wikimedia/wikipedia")
    p.add_argument("--config", default="20231101.ga")
    p.add_argument("--tokenizer", default="FacebookAI/xlm-roberta-large")
    p.add_argument("--streaming", action="store_true")
    p.add_argument("--seq_len", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--prefetch", type=int, default=4)
    p.add_argument("--shuffle_buffer", type=int, default=10000)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--compute_ms", type=float, default=0.0,
                   help="Simulated per-step compute (ms); reveals if data keeps up.")
    args = p.parse_args()

    print(f"dataset={args.dataset}:{args.config} streaming={args.streaming} "
          f"workers={args.num_workers} batch={args.batch_size}")
    loader = build_loader(args)
    it = iter(loader)

    for _ in range(args.warmup):
        next(it)

    n_tokens = 0
    n_samples = 0
    wait_times = []
    t_start = time.perf_counter()
    for _ in range(args.steps):
        t0 = time.perf_counter()
        batch = next(it)
        wait_times.append(time.perf_counter() - t0)
        n_samples += batch["input_ids"].shape[0]
        n_tokens += int(batch["attention_mask"].sum())
        if args.compute_ms:
            time.sleep(args.compute_ms / 1e3)
    elapsed = time.perf_counter() - t_start

    wait_times.sort()
    mean_wait = sum(wait_times) / len(wait_times)
    p50 = wait_times[len(wait_times) // 2]
    p95 = wait_times[int(len(wait_times) * 0.95)]
    pad_frac = 1 - n_tokens / (n_samples * args.seq_len)

    print(f"\n--- data throughput over {args.steps} steps ---")
    print(f"  samples/s         {n_samples / elapsed:10.1f}")
    print(f"  tokens/s (real)   {n_tokens / elapsed:10.0f}")
    print(f"  batch wait  mean  {mean_wait * 1e3:8.2f} ms")
    print(f"              p50   {p50 * 1e3:8.2f} ms")
    print(f"              p95   {p95 * 1e3:8.2f} ms")
    print(f"  padding fraction  {pad_frac * 100:8.1f}%  (wasted compute on PAD)")
    if args.compute_ms:
        stall = sum(max(0.0, w - args.compute_ms / 1e3) for w in wait_times) / args.steps
        print(f"  data-bound stall  {stall * 1e3:8.2f} ms/step at {args.compute_ms:.0f}ms compute")


if __name__ == "__main__":
    main()
