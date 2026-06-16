"""Export a vocab-pruned student as a standalone checkpoint + pruned tokenizer.

Training (train/vocab_pruning.py) shrinks the student embedding to keep_ids
and remaps full-vocab ids on the fly; the remap is saved as vocab_map.json.
This script bakes the remap into the tokenizer so inference needs no table:

- Unigram: the vocab list is filtered to the kept pieces, so token at new id i
  is the original token keep_ids[i] and the tokenizer emits pruned ids
  directly. Pieces that were dropped re-segment over the remaining vocab (or
  hit unk), which can differ from the training-time piece->unk remap; ids are
  exactly equivalent whenever every original piece is kept.
- BPE (mmBERT/Gemma): the vocab is pruned to the merge closure of keep_ids
  (every intermediate token of every merge that can produce a kept token,
  plus byte-fallback and added tokens; see train/vocab_pruning.merge_closure)
  and the merge table is filtered to merges fully inside the closed set.
  Every merge the original tokenizer ever applies while deriving a kept piece
  survives, so fully-kept text segments identically; other text re-segments
  into finer in-vocab pieces or byte fallback -- no id ever leaves the pruned
  range. Kept tokens get ids 0..len(keep_ids)-1 in keep_ids order (matching
  the pruned embedding rows). Training already merge-closes keep_ids, so
  normally the closure adds nothing here; if an older vocab_map.json without
  closure is exported, closure-only tokens get ids >= len(keep_ids) beyond
  the trained embedding and must be folded to unk (fold_oov) -- the export
  warns loudly and documents this in the emitted README.md.

Usage:
    uv run python scripts/export_student.py --checkpoint <dir> \
        --vocab_map <vocab_map.json> --tokenizer <name-or-path> --output <dir>
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from train.vocab_pruning import merge_closure_json  # noqa: E402

MODEL_FILE_PATTERNS = (
    "config.json",
    "generation_config.json",
    "*.safetensors",
    "*.safetensors.index.json",
    "pytorch_model*.bin",
)

VERIFY_SENTENCES = [
    "Hello world, this is a test sentence.",
    "Das ist ein etwas längerer deutscher Satz über Tokenisierung.",
    "Le modèle élève utilise un vocabulaire réduit.",
    "你好世界，这是一个测试。",
    "Numbers 12345 and symbols @#% mixed with emoji 😀.",
]


def fold_oov(ids, vocab_size: int, unk_id: int):
    """Map ids of pruned-away pieces (>= vocab_size, BPE export only) to unk."""
    if hasattr(ids, "masked_fill"):
        return ids.masked_fill(ids >= vocab_size, unk_id)
    return [i if i < vocab_size else unk_id for i in ids]


def _remap_post_processor(pp: dict, new_id: Dict[int, int]) -> None:
    if pp.get("type") == "Sequence":
        for sub in pp["processors"]:
            _remap_post_processor(sub, new_id)
        return
    if pp.get("type") == "TemplateProcessing":
        for st in pp["special_tokens"].values():
            st["ids"] = [new_id[i] for i in st["ids"]]
    for key in ("sep", "cls"):  # Bert/RobertaProcessing: [token, id]
        if isinstance(pp.get(key), list):
            pp[key][1] = new_id[pp[key][1]]


def prune_tokenizer_json(data: dict, keep_ids: Sequence[int], unk_id: int) -> dict:
    """Rewrite a tokenizers.json dict so kept token i emits id keep_ids.index(i)."""
    new_id = {orig: i for i, orig in enumerate(keep_ids)}
    if unk_id not in new_id:
        raise ValueError(f"unk_id {unk_id} must be in keep_ids")
    model = data["model"]
    added = data.get("added_tokens", [])

    if model["type"] == "Unigram":
        # sorted keep_ids put in-vocab ids before added-token-only ids, so the
        # filtered list position equals the pruned id
        if list(keep_ids) != sorted(set(keep_ids)):
            raise ValueError("Unigram pruning requires sorted, unique keep_ids")
        vocab: List = model["vocab"]
        added_ids = {t["id"] for t in added}
        for i in keep_ids:
            if i >= len(vocab) and i not in added_ids:
                raise ValueError(f"keep id {i} exceeds Unigram vocab")
        model["vocab"] = [vocab[i] for i in keep_ids if i < len(vocab)]
        model["unk_id"] = new_id[unk_id]
        data["added_tokens"] = [
            {**t, "id": new_id[t["id"]]} for t in added if t["id"] in new_id
        ]
    elif model["type"] == "BPE":
        vocab: Dict[str, int] = model["vocab"]
        all_ids = set(vocab.values()) | {t["id"] for t in added}
        missing = [i for i in keep_ids if i not in all_ids]
        if missing:
            raise ValueError(f"keep ids not in tokenizer vocab: {missing[:5]}")
        # kept tokens keep their pruned ids (matching the embedding rows);
        # closure-only tokens -- none when training already merge-closed
        # keep_ids -- get the ids after that
        perm = dict(new_id)
        for i in merge_closure_json(keep_ids, data):
            if i not in perm:
                perm[i] = len(perm)
        token_of = {i: t for t, i in vocab.items()}
        model["vocab"] = {token_of[i]: ni for i, ni in perm.items() if i in token_of}
        closed = set(model["vocab"])
        model["merges"] = [
            m for m in model["merges"]
            if (pair := m.split(" ", 1) if isinstance(m, str) else m)[0] in closed
            and pair[1] in closed and pair[0] + pair[1] in closed
        ]
        data["added_tokens"] = [{**t, "id": perm[t["id"]]} for t in added]
    else:
        raise ValueError(f"Unsupported tokenizer model type: {model['type']}")

    if data.get("post_processor"):
        _remap_post_processor(data["post_processor"], new_id)
    return data


def build_pruned_tokenizer(original, keep_ids: Sequence[int], unk_id: int):
    from tokenizers import Tokenizer
    from transformers import PreTrainedTokenizerFast

    if not getattr(original, "is_fast", False):
        raise ValueError("A fast tokenizer (tokenizers.json backend) is required")
    data = json.loads(original.backend_tokenizer.to_str())
    model_type = data["model"]["type"]
    pruned = prune_tokenizer_json(data, keep_ids, unk_id)
    backend = Tokenizer.from_str(json.dumps(pruned))
    specials = {
        name: getattr(original, name)
        for name in ("bos_token", "eos_token", "unk_token", "pad_token",
                     "cls_token", "sep_token", "mask_token")
        if getattr(original, name, None) is not None
    }
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        model_max_length=original.model_max_length,
        padding_side=original.padding_side,
        **specials,
    )
    return tokenizer, model_type


def verify(original, pruned, keep_ids: Sequence[int], unk_id: int,
           model_type: str, sentences: Sequence[str] = VERIFY_SENTENCES,
           fertility_text: Sequence[str] | None = None) -> bool:
    new_id = {orig: i for i, orig in enumerate(keep_ids)}
    total = len(pruned)  # pruned vocab incl. closure-only tokens (BPE)
    ok = True
    for text in sentences:
        ref_ids = original.encode(text)
        got = pruned.encode(text)
        if all(i in new_id for i in ref_ids):
            # fully-kept text must map to exactly the pruned ids, no fold
            if got == [new_id[i] for i in ref_ids]:
                status = "OK"
            else:
                status = f"FAIL ref={[new_id[i] for i in ref_ids]} got={got}"
                ok = False
        elif all(i < total for i in got):
            status = "OK (dropped pieces re-segmented in-vocab)"
        else:
            status = f"FAIL ids outside pruned vocab: {[i for i in got if i >= total]}"
            ok = False
        print(f"  [{status}] {text!r}")

    # fertility gate: pruning must not noticeably lengthen tokenizations of
    # TARGET-LANGUAGE text. A monolingual-pruned vocab is *expected* to
    # re-segment other languages, so generic multilingual sentences would show
    # a meaningless blow-up. With --fertility_text, measure on real target text
    # (where re-segmentation of OOV target words is the genuine signal);
    # otherwise fall back to the in-vocab subset of the verification sentences.
    if fertility_text:
        in_vocab = list(fertility_text)
    else:
        in_vocab = [t for t in sentences if all(i in new_id for i in original.encode(t))]
    if not in_vocab:
        print("  fertility: skipped (no in-vocab verification sentences; "
              "pass target-language --fertility_text to measure)")
        return ok
    words = sum(len(t.split()) for t in in_vocab) or 1
    orig_fertility = sum(len(original.encode(t)) for t in in_vocab) / words
    pruned_fertility = sum(len(pruned.encode(t)) for t in in_vocab) / words
    rel = pruned_fertility / orig_fertility - 1
    print(f"  fertility (on {len(in_vocab)} in-vocab sents): original {orig_fertility:.3f}, "
          f"pruned {pruned_fertility:.3f} ({rel:+.2%} tokens/word)")
    if rel > 0.02:
        print("  WARNING: pruned tokenizer fertility is >2% above the original - "
              "the kept vocabulary may be too small for this text distribution.")
    return ok


def export(checkpoint: Path, vocab_map: Path, tokenizer_name: str, output: Path,
           fertility_text: Path | None = None) -> bool:
    from transformers import AutoTokenizer

    with open(vocab_map) as f:
        vm = json.load(f)
    keep_ids, unk_id = vm["keep_ids"], vm["unk_id"]

    fertility_sents = None
    if fertility_text:
        fertility_sents = [l.strip() for l in Path(fertility_text).read_text().splitlines() if l.strip()]

    original = AutoTokenizer.from_pretrained(tokenizer_name)
    pruned, model_type = build_pruned_tokenizer(original, keep_ids, unk_id)
    print(f"Tokenizer model type: {model_type}; pruned vocab: {len(keep_ids)}")

    output.mkdir(parents=True, exist_ok=True)
    copied = []
    for entry in Path(checkpoint).iterdir():
        if any(fnmatch.fnmatch(entry.name, p) for p in MODEL_FILE_PATTERNS):
            shutil.copy2(entry, output / entry.name)
            copied.append(entry.name)
    if not any(n == "config.json" for n in copied):
        raise FileNotFoundError(f"No config.json in checkpoint {checkpoint}")
    pruned.save_pretrained(output)
    print(f"Exported to {output}: {sorted(copied)} + tokenizer files")

    print("Verifying id equivalence (original+remap vs pruned tokenizer):")
    ok = verify(original, pruned, keep_ids, unk_id, model_type, fertility_text=fertility_sents)
    print("Verification PASSED" if ok else "Verification FAILED")

    n_extra = len(pruned) - len(keep_ids)
    readme = _readme(model_type, len(keep_ids), n_extra, pruned.unk_token_id)
    (output / "README.md").write_text(readme)
    print("\n" + readme)
    return ok


def _readme(model_type: str, vocab_size: int, n_extra: int, unk_new: int) -> str:
    lines = [
        "# Vocab-pruned student export",
        "",
        f"Standalone checkpoint with a pruned {model_type} tokenizer "
        f"({vocab_size} kept tokens). Load and use like any HF model:",
        "",
        "```python",
        "from transformers import AutoModel, AutoTokenizer",
        "tokenizer = AutoTokenizer.from_pretrained(<this dir>)",
        "model = AutoModel.from_pretrained(<this dir>)",
        "outputs = model(**tokenizer(text, return_tensors='pt'))",
        "```",
        "",
        "The tokenizer emits pruned ids directly -- no remap table. Text whose",
        "original segmentation uses only kept tokens gets identical ids to the",
        "original tokenizer + training remap; other text re-segments into",
        "finer in-vocab pieces" + (
            " or byte fallback." if model_type == "BPE" else " or unk."),
    ]
    if n_extra > 0:
        lines += [
            "",
            "## WARNING: ids beyond the embedding",
            "",
            "This export was built from a keep set that was NOT merge-closed at",
            f"training time: {n_extra} merge-closure tokens got ids >= the model's",
            f"vocab_size ({vocab_size}) and have NO trained embedding row. Fold",
            "them to unk before the model:",
            "",
            "```python",
            f"ids.masked_fill_(ids >= {vocab_size}, {unk_new})  # unk_token_id",
            "```",
            "",
            "(see fold_oov in scripts/export_student.py)",
        ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vocab_map", type=Path, required=True)
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fertility_text", type=Path, default=None,
                        help="Optional file of target-language sentences (one per line) "
                             "to measure tokenizer fertility on.")
    args = parser.parse_args()
    sys.exit(0 if export(args.checkpoint, args.vocab_map, args.tokenizer, args.output,
                         args.fertility_text) else 1)


if __name__ == "__main__":
    main()
