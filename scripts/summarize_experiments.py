"""Join per-run efficiency.json + eval results into one comparison table.

    uv run python scripts/summarize_experiments.py --runs run1_moderngbert_relation run2_moderngbert_logit ...

Prints a table and writes results/summary.csv: one row per run with the active
distillation terms, training cost (wall-clock, tokens/s, peak VRAM, params), and
downstream accuracy/F1 on each German benchmark.
"""
import argparse
import csv
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATASETS = ("10kgnad", "germeval2018")


def load_json(path: Path):
    return json.loads(path.read_text()) if path.exists() else {}


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--out", default=str(REPO / "results" / "summary.csv"))
    args = p.parse_args()

    rows = []
    for run in args.runs:
        eff = load_json(REPO / "models" / run / "efficiency.json")
        row = {
            "run": run,
            "teacher": eff.get("teacher", ""),
            "logit_kd": eff.get("logit_kd_weight", 0),
            "repr": eff.get("repr_weight", 0),
            "params_M": eff.get("student_params_m", ""),
            "wall_s": eff.get("wall_clock_s", ""),
            "tok_per_s": eff.get("tokens_per_s", ""),
            "vram_GB": eff.get("peak_vram_gb", ""),
        }
        for ds in DATASETS:
            res = load_json(REPO / "results" / run / ds / "textcls_results.json")
            # german_textcls writes test accuracy + macro-f1; fall back across key names
            acc = res.get("test_accuracy", res.get("accuracy", ""))
            f1 = res.get("test_macro_f1", res.get("macro_f1", ""))
            row[f"{ds}_acc"] = round(acc, 4) if isinstance(acc, float) else acc
            row[f"{ds}_f1"] = round(f1, 4) if isinstance(f1, float) else f1
        rows.append(row)

    fields = list(rows[0].keys()) if rows else []
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    widths = {k: max(len(k), *(len(str(r[k])) for r in rows)) for k in fields}
    print("  ".join(k.ljust(widths[k]) for k in fields))
    for r in rows:
        print("  ".join(str(r[k]).ljust(widths[k]) for k in fields))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
