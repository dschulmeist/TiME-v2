"""Join the German benchmark suite outputs into one table.

Scans results/<name>/ for the three task outputs written by
run_german_benchmarks.sh and prints a combined table (and writes
results/german_summary.csv):

  text-cls : results/<name>/textcls_<ds>/textcls_results.json  (accuracy, macro-F1)
  NER      : results/<name>/ner.log                            ("Score:" = span-F1)
  UD       : results/<name>/ud/de_*.jsonl                      (UPOS, Lemmas, LAS)

Usage:
    uv run python evaluation/summarize_german_benchmarks.py [name ...]
    # no names -> every subdirectory of results/
"""
import csv
import glob
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"

COLUMNS = [
    "model", "10kgnad_acc", "10kgnad_f1", "germeval_acc", "germeval_f1",
    "ner_f1", "ud_upos", "ud_lemma", "ud_las",
]


def _textcls(name: str, ds: str):
    f = RESULTS / name / f"textcls_{ds}" / "textcls_results.json"
    if not f.exists():
        return None, None
    d = json.loads(f.read_text())
    return d.get("test_accuracy"), d.get("test_macro_f1")


def _ner(name: str):
    # prefer the structured result (seqeval span-F1) over grepping the log
    j = RESULTS / name / "ner" / "ner_results.json"
    if j.exists():
        try:
            return json.loads(j.read_text())["seqeval"]["overall_f1"]
        except (KeyError, ValueError):
            pass
    log = RESULTS / name / "ner.log"
    if log.exists():
        m = re.findall(r"Score:\s*([0-9.]+)", log.read_text())
        if m:
            return float(m[-1])
    return None


def _ud(name: str):
    files = sorted(glob.glob(str(RESULTS / name / "ud" / "de_*.jsonl")))
    if not files:
        return None, None, None
    # prefer the canonical de_hf.jsonl if present, else the first
    pick = next((f for f in files if f.endswith("de_hf.jsonl")), files[0])
    rec = json.loads(Path(pick).read_text())
    inner = next(iter(rec.values())) if rec else {}
    return inner.get("UPOS"), inner.get("Lemmas"), inner.get("LAS")


def row(name: str) -> dict:
    a10, f10 = _textcls(name, "10kgnad")
    ag, fg = _textcls(name, "germeval2018")
    upos, lemma, las = _ud(name)
    return {
        "model": name, "10kgnad_acc": a10, "10kgnad_f1": f10,
        "germeval_acc": ag, "germeval_f1": fg, "ner_f1": _ner(name),
        "ud_upos": upos, "ud_lemma": lemma, "ud_las": las,
    }


def main():
    if sys.argv[1:]:
        names = sys.argv[1:]
    elif RESULTS.exists():
        names = sorted(p.name for p in RESULTS.iterdir() if p.is_dir())
    else:
        names = []
    if not names:
        raise SystemExit("no results found under results/")
    rows = [row(n) for n in names]

    out = RESULTS / "german_summary.csv"
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)

    def fmt(v):
        return f"{v:.3f}" if isinstance(v, (int, float)) else "-"

    widths = {c: max(len(c), *(len(fmt(r[c]) if c != "model" else str(r[c])) for r in rows)) for c in COLUMNS}
    line = lambda cells: "  ".join(str(c).ljust(widths[col]) for col, c in zip(COLUMNS, cells))
    print(line(COLUMNS))
    for r in rows:
        print(line([r["model"]] + [fmt(r[c]) for c in COLUMNS[1:]]))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
