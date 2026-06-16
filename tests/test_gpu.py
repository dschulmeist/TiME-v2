"""GPU-only correctness tests. Skipped without CUDA."""
import pytest
import torch

from train.distiller import MiniLMDistiller
from train.losses import relation_kl

from conftest import tiny_bert

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

RELATIONS = {(1, 1): 1.0, (2, 2): 1.0, (3, 3): 1.0}


def make_qkv(seed, B=4, A_r=16, S=256, d_r=16, device="cuda", dtype=torch.float32):
    g = torch.Generator(device).manual_seed(seed)
    return tuple(
        torch.randn(B, A_r, S, d_r, generator=g, device=device, dtype=dtype)
        for _ in range(3)
    )


def ragged_mask(B=4, S=256, device="cuda"):
    lengths = torch.linspace(S // 4, S, B, dtype=torch.long)
    return (torch.arange(S)[None, :] < lengths[:, None]).long().to(device)


def cuda_distiller(seed=0, **kwargs):
    torch.manual_seed(seed)
    teacher = tiny_bert(hidden_size=32, num_layers=3, num_heads=4).eval()
    student = tiny_bert(hidden_size=16, num_layers=2, num_heads=2).eval()
    defaults = dict(L=2, M=2, relations=RELATIONS, A_r=4, teacher_dtype=torch.float32)
    defaults.update(kwargs)
    return MiniLMDistiller(teacher=teacher, student=student, **defaults).cuda()


def cuda_batch(B=4, S=32):
    torch.manual_seed(3)
    input_ids = torch.randint(1, 99, (B, S), device="cuda")
    mask = ragged_mask(B, S)
    input_ids[mask == 0] = 0
    return {"input_ids": input_ids, "attention_mask": mask}


def test_chunked_equals_unchunked_on_cuda():
    qkv_T, qkv_S = make_qkv(0), make_qkv(1)
    mask = ragged_mask()
    full = relation_kl(qkv_T, qkv_S, RELATIONS, mask, head_chunk_size=None)
    chunked = relation_kl(qkv_T, qkv_S, RELATIONS, mask, head_chunk_size=3)
    torch.testing.assert_close(chunked, full, rtol=1e-5, atol=1e-6)


def test_loss_stays_fp32_under_cuda_autocast():
    qkv_T, qkv_S = make_qkv(2), make_qkv(3)
    mask = ragged_mask()
    plain = relation_kl(qkv_T, qkv_S, RELATIONS, mask)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        autocast = relation_kl(qkv_T, qkv_S, RELATIONS, mask)
    torch.testing.assert_close(autocast, plain, rtol=0, atol=0)


def test_chunking_reduces_peak_memory():
    qkv_T, qkv_S = make_qkv(4, B=8, A_r=32, S=512), make_qkv(5, B=8, A_r=32, S=512)
    mask = ragged_mask(B=8, S=512)

    def peak(chunk):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        relation_kl(qkv_T, qkv_S, RELATIONS, mask, head_chunk_size=chunk)
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated()

    assert peak(4) < peak(None) / 2


def test_bf16_teacher_close_to_fp32_teacher():
    batch = cuda_batch()
    (fp32_loss,) = cuda_distiller(teacher_dtype=torch.float32)(**batch)
    (bf16_loss,) = cuda_distiller(teacher_dtype=torch.bfloat16)(**batch)
    torch.testing.assert_close(bf16_loss, fp32_loss, rtol=0.05, atol=1e-3)


def test_sdpa_and_eager_attention_record_same_qkv():
    batch = cuda_batch()
    losses = {}
    for impl in ("sdpa", "eager"):
        torch.manual_seed(0)
        teacher = tiny_bert(hidden_size=32, num_layers=3, num_heads=4).eval()
        torch.manual_seed(1)
        student = tiny_bert(hidden_size=16, num_layers=2, num_heads=2).eval()
        teacher.set_attn_implementation(impl)
        student.set_attn_implementation(impl)
        distiller = MiniLMDistiller(
            teacher=teacher, student=student, L=2, M=2,
            relations=RELATIONS, A_r=4, teacher_dtype=torch.float32,
        ).cuda()
        (losses[impl],) = distiller(**batch)
    torch.testing.assert_close(losses["sdpa"], losses["eager"], rtol=1e-4, atol=1e-5)


def test_pad_invariance_on_cuda_bf16():
    distiller = cuda_distiller(teacher_dtype=torch.bfloat16)
    batch = cuda_batch()
    (base,) = distiller(**batch)

    scrambled = batch["input_ids"].clone()
    pad = batch["attention_mask"] == 0
    scrambled[pad] = torch.randint(1, 99, (int(pad.sum()),), device="cuda")
    (loss,) = distiller(input_ids=scrambled, attention_mask=batch["attention_mask"])
    torch.testing.assert_close(loss, base, rtol=1e-3, atol=1e-4)


def test_compiled_student_matches_eager():
    batch = cuda_batch()
    (eager,) = cuda_distiller(seed=0)(**batch)
    torch.manual_seed(0)
    teacher = tiny_bert(hidden_size=32, num_layers=3, num_heads=4).eval()
    student = tiny_bert(hidden_size=16, num_layers=2, num_heads=2).eval()
    compiled = MiniLMDistiller(
        teacher=teacher, student=student, L=2, M=2, relations=RELATIONS,
        A_r=4, teacher_dtype=torch.float32, compile_student=True, compile_loss=True,
    ).cuda()
    (loss,) = compiled(**batch)
    loss.backward()
    torch.testing.assert_close(loss, eager, rtol=1e-4, atol=1e-5)
    assert any(p.grad is not None for p in student.parameters())


def test_training_step_with_8bit_adam():
    bnb = pytest.importorskip("bitsandbytes")
    distiller = cuda_distiller()
    distiller.student.train()
    optimizer = bnb.optim.AdamW8bit(distiller.student.parameters(), lr=5e-4)
    batch = cuda_batch()
    (first,) = distiller(**batch)
    first.backward()
    optimizer.step()
    optimizer.zero_grad()
    distiller.student.eval()
    (second,) = distiller(**batch)
    assert second.item() < first.item()
