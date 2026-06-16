import json

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers, processors, trainers
from transformers import AutoModel, AutoTokenizer, BertConfig, BertModel, PreTrainedTokenizerFast

from scripts.export_student import _readme, export, fold_oov, prune_tokenizer_json
from train.vocab_pruning import apply_vocab_pruning, merge_closure, merge_closure_json

CORPUS = [
    "hello world this is a test",
    "the quick brown fox jumps over the lazy dog",
    "tokenizers turn text into ids",
    "hello again world of tests",
]
COVERED = ["hello world this is a test", "the quick brown fox"]
OOV_TEXT = "hello zyxqzyxq world"  # rare pieces fall outside the kept set


def _tiny_bpe_dir(path):
    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Metaspace()
    trainer = trainers.BpeTrainer(
        vocab_size=120, special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"]
    )
    tok.train_from_iterator(CORPUS + [OOV_TEXT], trainer)
    tok.post_processor = processors.TemplateProcessing(
        single="<bos> $A <eos>",
        special_tokens=[("<bos>", tok.token_to_id("<bos>")), ("<eos>", tok.token_to_id("<eos>"))],
    )
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok, pad_token="<pad>", unk_token="<unk>",
        bos_token="<bos>", eos_token="<eos>",
    )
    fast.save_pretrained(path)
    return fast


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    root = tmp_path_factory.mktemp("export")
    tok_dir, ckpt_dir, out_dir = root / "tok", root / "ckpt", root / "out"
    original = _tiny_bpe_dir(tok_dir)

    vocab_size = original.backend_tokenizer.get_vocab_size()
    keep = set(original.all_special_ids)
    for text in COVERED:
        keep.update(original.encode(text))
    # as training does: merge-close so the embedding covers every emitted id
    keep_ids = merge_closure(sorted(keep), original)
    unk_id = original.unk_token_id

    torch.manual_seed(0)
    student = BertModel(BertConfig(
        vocab_size=vocab_size, hidden_size=16, num_hidden_layers=2,
        num_attention_heads=2, intermediate_size=64, max_position_embeddings=64,
        pad_token_id=original.pad_token_id,
    ))
    apply_vocab_pruning(student, keep_ids, unk_id)
    student.save_pretrained(ckpt_dir)
    with open(ckpt_dir / "vocab_map.json", "w") as f:
        json.dump({"keep_ids": keep_ids, "unk_id": unk_id}, f)

    ok = export(ckpt_dir, ckpt_dir / "vocab_map.json", str(tok_dir), out_dir)
    return {
        "ok": ok, "out_dir": out_dir, "original": original,
        "keep_ids": keep_ids, "unk_id": unk_id,
    }


def test_export_verification_passes(exported):
    assert exported["ok"]


def test_covered_ids_identical(exported):
    pruned = AutoTokenizer.from_pretrained(exported["out_dir"])
    new_id = {orig: i for i, orig in enumerate(exported["keep_ids"])}
    for text in COVERED:
        ref = [new_id[i] for i in exported["original"].encode(text)]
        assert pruned.encode(text) == ref  # direct, no fold needed


def test_oov_stays_in_pruned_vocab(exported):
    pruned = AutoTokenizer.from_pretrained(exported["out_dir"])
    K = len(exported["keep_ids"])
    raw = pruned.encode(OOV_TEXT)
    assert all(i < K for i in raw)  # no id ever leaves the pruned range
    assert len(raw) >= len(exported["original"].encode(OOV_TEXT))  # finer or unk


def test_exported_model_roundtrip(exported):
    pruned = AutoTokenizer.from_pretrained(exported["out_dir"])
    model = AutoModel.from_pretrained(exported["out_dir"])
    K = len(exported["keep_ids"])
    assert model.config.vocab_size == K
    assert model.embeddings.word_embeddings.num_embeddings == K
    for text in (COVERED[0], OOV_TEXT):  # no fold needed, even on oov text
        batch = pruned(text, return_tensors="pt")
        out = model(**batch)
        assert out.last_hidden_state.shape[1] == batch["input_ids"].shape[1]


def test_readme_written(exported):
    readme = (exported["out_dir"] / "README.md").read_text()
    assert "AutoTokenizer.from_pretrained" in readme
    assert "WARNING" not in readme  # keep set was merge-closed, no fold needed
    assert "WARNING" in _readme("BPE", 100, n_extra=3, unk_new=1)


def test_fold_oov_tensor_and_list():
    assert fold_oov([0, 3, 7, 2], vocab_size=4, unk_id=1) == [0, 3, 1, 2]
    folded = fold_oov(torch.tensor([[0, 5, 2]]), vocab_size=4, unk_id=1)
    assert folded.tolist() == [[0, 1, 2]]


def test_merge_closure_adds_intermediates(tmp_path):
    original = _tiny_bpe_dir(tmp_path / "tok")
    data = json.loads(original.backend_tokenizer.to_str())
    vocab, merges = data["model"]["vocab"], data["model"]["merges"]
    # pick a merged token and keep only it: closure must pull in its operands
    pair = next(m for m in reversed(merges) if (m[0] + m[1]) in vocab)
    target = pair[0] + pair[1]
    closed = merge_closure_json([vocab[target]], data)
    assert vocab[target] in closed
    for m in merges:
        if m[0] + m[1] == target:
            assert vocab[m[0]] in closed and vocab[m[1]] in closed
    # closure is idempotent and sorted
    assert merge_closure_json(closed, data) == closed == sorted(closed)
    # tokenizer-object wrapper agrees, and always includes added tokens
    assert set(merge_closure([vocab[target]], original)) >= set(
        closed) | set(original.all_special_ids)


def test_merge_closure_non_bpe_passthrough(tmp_path):
    pieces = [("<unk>", 0.0), ("▁a", -1.0), ("▁b", -1.0)]
    tok = Tokenizer(models.Unigram(pieces, unk_id=0))
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="<unk>")
    assert merge_closure([2, 0, 2], fast) == [0, 2]


def test_legacy_unclosed_keep_set_needs_fold(tmp_path):
    """keep_ids without closure: closure-only tokens get tail ids past the
    embedding; fold_oov remains the guarded escape hatch."""
    original = _tiny_bpe_dir(tmp_path / "tok")
    data = json.loads(original.backend_tokenizer.to_str())
    keep = sorted(set(original.all_special_ids) | set(original.encode(COVERED[0])))
    unk_id = original.unk_token_id
    closed = merge_closure_json(keep, data)
    assert len(closed) > len(keep)
    pruned = Tokenizer.from_str(json.dumps(prune_tokenizer_json(data, keep, unk_id)))
    assert pruned.get_vocab_size() == len(closed)

    new_id = {orig: i for i, orig in enumerate(keep)}
    K, unk_new = len(keep), new_id[unk_id]
    # covered text: exact pruned ids, all below K
    ref = [new_id[i] for i in original.encode(COVERED[0], add_special_tokens=False)]
    assert pruned.encode(COVERED[0], add_special_tokens=False).ids == ref
    # other text may emit closure-only tail ids; fold maps them to unk
    for text in CORPUS + [OOV_TEXT]:
        ids = pruned.encode(text, add_special_tokens=False).ids
        assert all(i < len(closed) for i in ids)
        assert all(i < K for i in fold_oov(ids, K, unk_new))


def test_unigram_pruning_direct():
    pieces = [
        ("<pad>", 0.0), ("<unk>", 0.0),
        ("▁hello", -1.0), ("▁world", -1.0), ("▁foo", -2.0),
        ("▁", -3.0), ("h", -4.0), ("o", -4.0), ("w", -4.0),
    ]
    tok = Tokenizer(models.Unigram(pieces, unk_id=1))
    tok.pre_tokenizer = pre_tokenizers.Metaspace()
    data = json.loads(tok.to_str())
    keep_ids = [0, 1, 2, 3, 5, 6, 7]  # drop "▁foo" (4) and "w" (8)
    pruned = Tokenizer.from_str(json.dumps(prune_tokenizer_json(data, keep_ids, unk_id=1)))

    new_id = {orig: i for i, orig in enumerate(keep_ids)}
    text = "hello world"  # all original pieces kept
    ref = [new_id[i] for i in tok.encode(text).ids]
    assert pruned.encode(text).ids == ref
    assert all(i < len(keep_ids) for i in pruned.encode("hello foo wow").ids)


def test_mmbert_bpe_pruning():
    try:
        original = AutoTokenizer.from_pretrained("jhu-clsp/mmBERT-base")
    except OSError:
        pytest.skip("mmBERT tokenizer unavailable (offline)")
    data = json.loads(original.backend_tokenizer.to_str())
    assert data["model"]["type"] == "BPE"

    text = "Hello world, das ist ein Test."
    keep = set(original.all_special_ids) | set(original.encode(text)) | set(range(512))
    keep_ids = merge_closure_json(sorted(keep), data)
    unk_id = original.unk_token_id
    pruned = Tokenizer.from_str(json.dumps(prune_tokenizer_json(data, keep_ids, unk_id)))

    new_id = {orig: i for i, orig in enumerate(keep_ids)}
    K = len(new_id)
    assert pruned.get_vocab_size() == K  # closed keep set: no tail ids
    # fully-kept text: identical segmentation and ids, no fold
    ref = [new_id[i] for i in original.encode(text)]
    assert pruned.encode(text).ids == ref
    # adversarial text re-segments into in-vocab pieces (byte fallback incl.)
    for t in ["你好世界 " + text, "Ψυχή edge 😀 cases", "ᚠᚢᚦᚨᚱᚲ ⵜⴰⵎⴰⵣⵉⵖⵜ"]:
        ids = pruned.encode(t).ids
        assert all(i < K for i in ids)
        assert pruned.decode(ids, skip_special_tokens=True).strip() == t
