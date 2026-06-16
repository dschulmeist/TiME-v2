"""Pre-tokenize and pre-pack a (small) language corpus to memory-mapped Arrow.

For single-shard / low-resource languages, streaming serializes tokenize+pack
onto one worker and re-does it every epoch (measured: ~5x GPU starvation on
Irish). Materializing once lets training memory-map the packed rows and
tokenize-pack nothing in the hot loop - the regime where the dataloader
sustains ~4000 samples/s with zero stall.

Output columns match train/packing.py (input_ids, attention_mask, segment_ids,
position_ids), so training loads it directly with the default collator. Stored
ids are the FULL teacher vocab; vocab pruning still happens at train time and
the distiller remaps the student - the two are orthogonal.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train.data_pipeline import _TokenizeFn  # noqa: E402
from train.packing import PackFn  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--dataset", required=True)
    p.add_argument("--config", default=None)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max_seq_len", type=int, default=512)
    p.add_argument("--text_column", default="text")
    p.add_argument("--max_docs", type=int, default=0, help="0 = whole corpus.")
    p.add_argument("--num_proc", type=int, default=8)
    p.add_argument("--map_batch_size", type=int, default=1000)
    args = p.parse_args()

    from datasets import load_dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    t0 = time.time()

    # download to the local arrow cache (map-style) so .map can use num_proc;
    # streaming would force single-process. small langs fit on disk by premise.
    ds = load_dataset(args.dataset, args.config, split="train")
    if args.max_docs:
        ds = ds.select(range(min(args.max_docs, len(ds))))
    print(f"loaded {len(ds):,} docs in {time.time()-t0:.0f}s")

    tokenized = ds.map(
        _TokenizeFn(tokenizer, args.text_column, args.max_seq_len),
        batched=True, batch_size=args.map_batch_size, num_proc=args.num_proc,
        remove_columns=ds.column_names, desc="tokenize",
    )
    packed = tokenized.map(
        PackFn(args.max_seq_len, tokenizer.pad_token_id),
        batched=True, batch_size=args.map_batch_size, num_proc=args.num_proc,
        remove_columns=tokenized.column_names, desc="pack",
    )
    packed.save_to_disk(str(args.output))

    real = sum(sum(r) for r in packed["attention_mask"][:2000])
    pad_frac = 1 - real / (min(2000, len(packed)) * args.max_seq_len)
    print(f"packed {len(packed):,} rows of {args.max_seq_len} tok in {time.time()-t0:.0f}s total")
    print(f"  -> {args.output}  (padding fraction ~{pad_frac:.1%})")


if __name__ == "__main__":
    main()
