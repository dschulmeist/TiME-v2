"""Rank a run's student checkpoints by distillation loss on unseen text.

Handles vocab-pruned students via the run's vocab_map.json (the distiller
remaps ids for the student; the teacher keeps the full vocabulary).
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train.distiller import MiniLMDistiller  # noqa: E402


FLORES_CODES = {"ga": "gle_Latn"}
WIKI_CODES = {"ga": "20231101.ga"}


def eval_texts(lang: str, n: int = 1000):
    """FLORES+ dev if accessible, else held-out Wikipedia (out-of-domain for
    FineWeb-trained runs either way - both are unseen text)."""
    from datasets import load_dataset

    try:
        ds = load_dataset("openlanguagedata/flores_plus", FLORES_CODES[lang], split="dev")
        return [r["text"] for r in ds]
    except Exception as e:
        print(f"flores_plus unavailable ({str(e)[:80]}); falling back to Wikipedia")
    ds = load_dataset("wikimedia/wikipedia", WIKI_CODES[lang], split="train", streaming=True)
    import itertools
    return [r["text"] for r in itertools.islice(iter(ds), n)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run_dir", required=True, help="Training output dir (contains student/, vocab_map.json)")
    p.add_argument("--teacher", required=True)
    p.add_argument("--L", type=int, required=True)
    p.add_argument("--num_relation_heads", type=int, required=True)
    p.add_argument("--lang", default="ga")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--out_csv", default=None)
    args = p.parse_args()

    from transformers import AutoModel, AutoTokenizer

    run = Path(args.run_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.teacher)
    teacher = AutoModel.from_pretrained(args.teacher, attn_implementation="sdpa").eval()

    remap = None
    vocab_map = run / "vocab_map.json"
    if vocab_map.exists():
        keep = json.loads(vocab_map.read_text())
        remap = torch.full((len(tokenizer),), keep["keep_ids"].index(keep["unk_id"]), dtype=torch.long)
        remap[torch.tensor(keep["keep_ids"])] = torch.arange(len(keep["keep_ids"]))

    texts = eval_texts(args.lang)
    print(f"{len(texts)} eval sentences ({args.lang})")
    batches = []
    for i in range(0, len(texts), args.batch_size):
        batches.append(tokenizer(
            texts[i:i + args.batch_size], return_tensors="pt", padding=True,
            truncation=True, max_length=args.max_length,
        ))

    checkpoints = sorted(run.glob("student/checkpoint-*"), key=lambda c: int(c.name.split("-")[-1]))
    relations = {(1, 1): 1.0, (2, 2): 1.0, (3, 3): 1.0}
    results = []
    for ckpt in checkpoints:
        student = AutoModel.from_pretrained(ckpt).eval()
        distiller = MiniLMDistiller(
            teacher=teacher, student=student, L=args.L,
            M=student.config.num_hidden_layers, relations=relations,
            A_r=args.num_relation_heads, truncate_teacher=False,
            student_input_remap=remap,
        ).to(device).eval()
        total, n = 0.0, 0
        with torch.no_grad():
            for enc in batches:
                ids = enc["input_ids"].to(device)
                mask = enc["attention_mask"].to(device)
                (loss,) = distiller(input_ids=ids, attention_mask=mask)
                total += loss.item()
                n += 1
        step = int(ckpt.name.split("-")[-1])
        results.append((step, total / n))
        print(f"checkpoint-{step}: val_loss {total / n:.4f}", flush=True)
        distiller.teacher_recorder.close()
        distiller.student_recorder.close()
        del distiller, student
        if device == "cuda":
            torch.cuda.empty_cache()

    best = min(results, key=lambda r: r[1])
    print(f"\nBEST: checkpoint-{best[0]} (val_loss {best[1]:.4f})")
    if args.out_csv:
        with open(args.out_csv, "w") as f:
            f.write("step,val_loss\n")
            f.writelines(f"{s},{l}\n" for s, l in results)


if __name__ == "__main__":
    main()
