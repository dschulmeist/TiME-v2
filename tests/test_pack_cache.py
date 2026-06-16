import datasets
import torch
from transformers import default_data_collator

from train.data_pipeline import _TokenizeFn
from train.packing import PackFn


class _Tok:
    """Tiny deterministic word tokenizer mimicking the HF fast-tokenizer API
    surface the build script uses."""
    pad_token_id = 0

    def __call__(self, texts, truncation=True, padding=False, max_length=512,
                 return_token_type_ids=True, add_special_tokens=True):
        out = [[(hash(w) % 90) + 5 for w in t.split()][:max_length] for t in texts]
        return {"input_ids": out, "attention_mask": [[1] * len(x) for x in out]}


def test_pack_cache_roundtrip_and_collation(tmp_path):
    """The offline pack-cache path (tokenize -> pack -> save_to_disk -> load ->
    collate) must yield exactly the columns the distiller consumes, batched."""
    docs = {"text": [f"word{i} " * (i % 11 + 3) for i in range(200)]}
    ds = datasets.Dataset.from_dict(docs)
    tok = _Tok()

    tokenized = ds.map(_TokenizeFn(tok, "text", 32), batched=True,
                       remove_columns=ds.column_names)
    packed = tokenized.map(PackFn(32, tok.pad_token_id), batched=True,
                           remove_columns=tokenized.column_names)
    packed.save_to_disk(str(tmp_path / "cache"))

    loaded = datasets.load_from_disk(str(tmp_path / "cache")).with_format("torch")
    assert set(loaded.column_names) == {"input_ids", "attention_mask", "segment_ids", "position_ids"}
    assert all(loaded[0][c].shape == (32,) for c in loaded.column_names)

    batch = default_data_collator([loaded[i] for i in range(8)])
    assert batch["input_ids"].shape == (8, 32)
    assert batch["segment_ids"].shape == (8, 32)
    # within a row, position_ids restart at 0 at each new segment
    seg, pos = loaded[0]["segment_ids"], loaded[0]["position_ids"]
    real = loaded[0]["attention_mask"].bool()
    boundaries = (seg[1:] != seg[:-1]) & real[1:]
    assert (pos[1:][boundaries] == 0).all()


def test_pack_cache_preserves_tokens(tmp_path):
    docs = {"text": [f"a{i} b{i} c{i}" for i in range(50)]}
    ds = datasets.Dataset.from_dict(docs)
    tok = _Tok()
    tokenized = ds.map(_TokenizeFn(tok, "text", 16), batched=True, remove_columns=ds.column_names)
    flat_in = [t for row in tokenized["input_ids"] for t in row]
    packed = tokenized.map(PackFn(16, tok.pad_token_id), batched=True,
                           remove_columns=tokenized.column_names)
    flat_packed = [t for row, m in zip(packed["input_ids"], packed["attention_mask"])
                   for t, keep in zip(row, m) if keep]
    assert flat_packed == flat_in
