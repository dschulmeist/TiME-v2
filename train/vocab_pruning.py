"""Training-time vocabulary pruning for student models.

Multilingual tokenizers (XLM-R: 250k, mmBERT/Gemma: 256k) make the embedding
table the bulk of a tiny student's parameters, while a single language uses
only a small fraction of the ids. The student is built with a pruned embedding
table over the ids that actually occur in the training corpus; the distiller
remaps input ids for the student on the fly. The teacher keeps the full
vocabulary - the relation loss never touches vocab space, so distillation is
unaffected.
"""
from __future__ import annotations

import itertools
import json
import logging
from collections import Counter, defaultdict
from typing import Iterable, List, Sequence

import torch
from torch import nn

logger = logging.getLogger(__name__)


def count_token_ids(tokenized_examples: Iterable[dict], num_docs: int) -> Counter:
    """Count token-id occurrences over the first `num_docs` tokenized examples."""
    counts: Counter = Counter()
    for example in itertools.islice(tokenized_examples, num_docs):
        ids = example["input_ids"]
        counts.update(ids.tolist() if hasattr(ids, "tolist") else ids)
    return counts


def select_vocab(counts: Counter, keep_size: int, must_keep: Sequence[int]) -> List[int]:
    """The `keep_size` most frequent ids, plus `must_keep` (special tokens),
    returned sorted. Coverage of the sampled corpus is logged."""
    kept = set(must_keep)
    for token_id, _ in counts.most_common():
        if len(kept) >= keep_size:
            break
        kept.add(token_id)
    total = sum(counts.values())
    covered = sum(c for t, c in counts.items() if t in kept)
    logger.info(
        "Vocab pruning: keeping %d/%d ids, %.3f%% corpus coverage",
        len(kept), keep_size, 100 * covered / max(total, 1),
    )
    return sorted(kept)


def select_vocab_by_coverage(counts: Counter, coverage: float, must_keep: Sequence[int]) -> List[int]:
    """The smallest most-frequent id set covering `coverage` of corpus tokens,
    plus `must_keep`. 0.99+ keeps tokenizer fertility essentially unchanged;
    lower thresholds trade size for fertility."""
    if not 0 < coverage <= 1:
        raise ValueError(f"coverage must be in (0, 1], got {coverage}")
    kept = set(must_keep)
    total = sum(counts.values())
    covered = sum(c for t, c in counts.items() if t in kept)
    for token_id, count in counts.most_common():
        if covered >= coverage * total:
            break
        if token_id not in kept:
            kept.add(token_id)
            covered += count
    logger.info(
        "Vocab pruning (coverage %.4f): keeping %d ids, %.3f%% corpus coverage",
        coverage, len(kept), 100 * covered / max(total, 1),
    )
    return sorted(kept)


def merge_closure(keep_ids: Sequence[int], tokenizer) -> List[int]:
    """Expand `keep_ids` to its BPE merge closure, returned sorted.

    For every kept token, every intermediate token of every merge derivation
    that can produce it is added, plus all byte-fallback tokens (<0xNN>) and
    all added/special tokens. A BPE tokenizer pruned to a merge-closed set
    (with merges filtered accordingly) segments fully-covered text identically
    to the original and never emits an id outside the closed set, so the
    student embedding must cover the closure. Non-BPE tokenizers are returned
    unchanged (Unigram needs no closure: dropped pieces re-segment).
    """
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("A fast tokenizer (tokenizers.json backend) is required")
    return merge_closure_json(keep_ids, json.loads(tokenizer.backend_tokenizer.to_str()))


def merge_closure_json(keep_ids: Sequence[int], data: dict) -> List[int]:
    """`merge_closure` over an already-parsed tokenizers.json dict."""
    if data["model"]["type"] != "BPE":
        return sorted(dict.fromkeys(keep_ids))

    vocab = data["model"]["vocab"]
    producers = defaultdict(list)
    for merge in data["model"]["merges"]:
        left, right = merge.split(" ", 1) if isinstance(merge, str) else merge
        producers[left + right].append((left, right))

    id_of = dict(vocab)
    tok_of = {i: t for t, i in vocab.items()}
    closed_ids = set(keep_ids)
    closed_ids.update(t["id"] for t in data.get("added_tokens", []))
    closed_ids.update(
        i for t, i in vocab.items()
        if len(t) == 6 and t.startswith("<0x") and t.endswith(">")
    )
    stack = [tok_of[i] for i in closed_ids if i in tok_of]
    seen = set(stack)
    while stack:
        for operand in itertools.chain.from_iterable(producers.get(stack.pop(), ())):
            if operand not in seen:
                seen.add(operand)
                closed_ids.add(id_of[operand])
                stack.append(operand)

    added = len(closed_ids) - len(set(keep_ids))
    logger.info("Merge closure: %d keep ids -> %d (+%d)", len(set(keep_ids)),
                len(closed_ids), added)
    return sorted(closed_ids)


def _embedding_module(model: nn.Module) -> tuple[nn.Module, str]:
    embeddings = model.embeddings
    for attr in ("word_embeddings", "tok_embeddings"):
        if hasattr(embeddings, attr):
            return embeddings, attr
    raise ValueError(f"Unsupported embedding layout on {type(model).__name__}")


def apply_vocab_pruning(student: nn.Module, keep_ids: Sequence[int], unk_id: int) -> torch.Tensor:
    """Shrink the student's embedding table to `keep_ids` and return the remap.

    The remap is a LongTensor over the original vocab: kept ids map to their
    new position, everything else to `unk_id`'s new position. `unk_id` and the
    pad id must be in `keep_ids`.
    """
    keep = list(dict.fromkeys(keep_ids))
    if unk_id not in keep:
        raise ValueError(f"unk_id {unk_id} must be in keep_ids")
    pad_id = student.config.pad_token_id
    if pad_id is not None and pad_id not in keep:
        raise ValueError(f"pad_token_id {pad_id} must be in keep_ids")

    container, attr = _embedding_module(student)
    old: nn.Embedding = getattr(container, attr)
    index = torch.tensor(keep, dtype=torch.long)
    new_pad = keep.index(pad_id) if pad_id is not None else None
    new = nn.Embedding(len(keep), old.embedding_dim, padding_idx=new_pad)
    with torch.no_grad():
        new.weight.copy_(old.weight[index])
    setattr(container, attr, new)

    remap = torch.full((student.config.vocab_size,), keep.index(unk_id), dtype=torch.long)
    remap[index] = torch.arange(len(keep))

    student.config.vocab_size = len(keep)
    if pad_id is not None:
        student.config.pad_token_id = new_pad
    logger.info(
        "Student embedding pruned: %d -> %d rows (%.1fM -> %.1fM params)",
        old.num_embeddings, len(keep),
        old.num_embeddings * old.embedding_dim / 1e6,
        len(keep) * old.embedding_dim / 1e6,
    )
    return remap
