"""Sequence packing: concatenate tokenized documents into fixed-length rows.

Packed rows carry `segment_ids` (which document each token belongs to) and
`position_ids` (restarting at 0 per document). Models receive a block-diagonal
attention mask and per-segment positions, so each document is processed exactly
as if it were alone in the batch - verified by tests - while the padding
fraction drops to near zero and shapes become fixed (torch.compile-friendly).
"""
from __future__ import annotations

from typing import Dict, List

PACK_COLUMNS = ("input_ids", "attention_mask", "segment_ids", "position_ids")


class PackFn:
    """Batched `.map()` callable that greedily packs examples into rows of
    exactly `max_seq_len` tokens. The final partial row is padded, so no data
    is dropped within a map batch.
    """

    def __init__(self, max_seq_len: int, pad_token_id: int):
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id

    def __call__(self, batch: Dict[str, List[List[int]]]) -> Dict[str, List[List[int]]]:
        rows: Dict[str, List[List[int]]] = {col: [] for col in PACK_COLUMNS}
        ids: List[int] = []
        segments: List[int] = []
        positions: List[int] = []
        segment = 0

        def flush() -> None:
            if not ids:
                return
            pad = self.max_seq_len - len(ids)
            rows["input_ids"].append(ids + [self.pad_token_id] * pad)
            rows["attention_mask"].append([1] * len(ids) + [0] * pad)
            rows["segment_ids"].append(segments + [-1] * pad)
            rows["position_ids"].append(positions + [0] * pad)
            ids.clear(), segments.clear(), positions.clear()

        for doc in batch["input_ids"]:
            doc = doc[: self.max_seq_len]
            if hasattr(doc, "tolist"):  # torch-formatted datasets yield tensors
                doc = doc.tolist()
            if len(ids) + len(doc) > self.max_seq_len:
                flush()
            ids.extend(doc)
            segments.extend([segment] * len(doc))
            positions.extend(range(len(doc)))
            segment += 1
        flush()
        return rows


def pack_dataset(tokenized_dataset, max_seq_len: int, pad_token_id: int, map_batch_size: int = 1000):
    """Apply packing to a tokenized (streaming or map-style) dataset."""
    drop = tokenized_dataset.column_names
    if drop is None:  # streaming datasets may not expose column names
        drop = ["input_ids", "attention_mask", "token_type_ids"]
    return tokenized_dataset.map(
        PackFn(max_seq_len, pad_token_id),
        batched=True,
        batch_size=map_batch_size,
        remove_columns=drop,
    )
