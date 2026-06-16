import pytest
import torch
from transformers import BertConfig, BertModel

VOCAB = 99
PAD_ID = 0


def tiny_bert(hidden_size: int, num_layers: int, num_heads: int) -> BertModel:
    config = BertConfig(
        vocab_size=VOCAB,
        hidden_size=hidden_size,
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
        intermediate_size=hidden_size * 4,
        max_position_embeddings=64,
        pad_token_id=PAD_ID,
    )
    return BertModel(config)


@pytest.fixture
def teacher():
    torch.manual_seed(0)
    return tiny_bert(hidden_size=32, num_layers=3, num_heads=4).eval()


@pytest.fixture
def student():
    torch.manual_seed(1)
    # eval mode for deterministic tests (no dropout); the Trainer flips this
    # to train mode during real runs
    return tiny_bert(hidden_size=16, num_layers=2, num_heads=2).eval()


@pytest.fixture
def batch():
    """A small batch with ragged padding (lengths 8, 5, 3)."""
    torch.manual_seed(2)
    B, S = 3, 8
    lengths = [8, 5, 3]
    input_ids = torch.randint(1, VOCAB, (B, S))
    attention_mask = torch.zeros(B, S, dtype=torch.long)
    for row, length in enumerate(lengths):
        attention_mask[row, :length] = 1
        input_ids[row, length:] = PAD_ID
    return {"input_ids": input_ids, "attention_mask": attention_mask}
