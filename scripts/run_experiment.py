"""Run one distillation experiment from a YAML config.

A user-friendly layer over `train.distillation`: a single readable config file
captures the whole setup, expands to the module's CLI, and runs it (single- or
multi-GPU via torchrun). Anything in the YAML can still be overridden on the
command line, e.g.:

    uv run python scripts/run_experiment.py experiments/run1_moderngbert_relation.yaml
    uv run python scripts/run_experiment.py experiments/run1_moderngbert_relation.yaml training.max_steps=2000

Config schema (all sections optional except teacher/student/output):

    teacher: <hf model id or path>           # distillation teacher
    student_architecture: <hf id>            # arch template for the student (default: BERT)
    output: <dir>
    data:
      dataset: <hf id>                       # OR pretokenized: <pack-cache dir>
      config: <subset>
      pretokenized: <dir>                    # pre-packed cache (build_pack_cache.py); skips streaming
      max_seq_len: 512
      pack: true                             # pack on the fly (ignored if pretokenized)
    student: {hidden, layers, heads, intermediate, L, A_r, prune_vocab, prune_vocab_coverage}
    loss:  {relations, rope_qk_weight, repr_weight, logit_kd_weight, logit_kd_temperature}
    training: {batch, grad_accum, max_steps, lr, warmup, save_steps, save_total_limit,
               seed, num_workers, compile, compile_student, no_truncate_teacher,
               teacher_dtype, head_chunk_size}
"""
import subprocess
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]


def _set(d, dotted, value):
    cur = d
    *parents, leaf = dotted.split(".")
    for p in parents:
        cur = cur.setdefault(p, {})
    # coerce simple scalars
    for cast in (int, float):
        try:
            value = cast(value); break
        except ValueError:
            pass
    if value in ("true", "false"):
        value = value == "true"
    cur[leaf] = value


def build_argv(cfg: dict) -> list[str]:
    data = cfg.get("data", {})
    st = cfg.get("student", {})
    loss = cfg.get("loss", {})
    tr = cfg.get("training", {})

    if "teacher" not in cfg or "output" not in cfg:
        raise SystemExit("config must set 'teacher' and 'output'")
    batch, accum = tr.get("batch", 64), tr.get("grad_accum", 1)
    workers = tr.get("num_workers", 8)

    data_args = ["data_params", "--max_seq_len", str(data.get("max_seq_len", 512))]
    if data.get("pretokenized"):
        data_args += ["--pretokenized_dataset_path", data["pretokenized"]]
    else:
        data_args += ["--dataset_name", data["dataset"]]
        if data.get("config"):
            data_args += ["--dataset_config_name", data["config"]]
        if data.get("pack", True):
            data_args += ["--pack"]

    train_args = [
        "training_params",
        "--per_device_train_batch_size", str(batch),
        "--gradient_accumulation_steps", str(accum),
        "--learning_rate", str(tr.get("lr", 6e-4)),
        "--max_steps", str(tr.get("max_steps", 15000)),
        "--warmup_steps", str(tr.get("warmup", 1500)),
        "--save_strategy", "steps",
        "--save_steps", str(tr.get("save_steps", 2500)),
        "--save_total_limit", str(tr.get("save_total_limit", 8)),
        "--logging_steps", str(tr.get("logging_steps", 100)),
        "--bf16", "true",
        "--max_grad_norm", "1.0",
        "--output_dir", cfg["output"],
        "--seed", str(tr.get("seed", 21)),
        "--dataloader_num_workers", str(workers),
        "--report_to", "none",
    ]
    if workers > 0:  # prefetch_factor is only valid with worker processes
        train_args += ["--dataloader_prefetch_factor", str(tr.get("prefetch_factor", 2))]

    # resume from the latest Trainer checkpoint if the run was interrupted.
    # These are the optimizer-bearing checkpoint-N dirs in the output root
    # (distinct from student/).
    import glob
    # resolve the output dir against the repo root so resume works regardless of
    # the launcher's cwd (the distillation subprocess runs with cwd=REPO too)
    out_dir = cfg["output"] if Path(cfg["output"]).is_absolute() else REPO / cfg["output"]
    ckpts = glob.glob(f"{out_dir}/checkpoint-*")
    if ckpts:
        latest = max(ckpts, key=lambda p: int(p.rsplit("-", 1)[-1]))
        train_args += ["--resume_from_checkpoint", latest]

    model_args = [
        "model_params",
        "--input_model_dir", cfg["teacher"],
        "--student_hidden_size", str(st.get("hidden", 384)),
        "--student_num_layers", str(st.get("layers", 6)),
        "--student_attention_heads", str(st.get("heads", 6)),
        "--L", str(st["L"]),
        "--num_relation_heads", str(st.get("A_r", 16)),
        "--minilm_relations", loss.get("relations", "{(1,1):1,(2,2):1,(3,3):1}"),
    ]
    if cfg.get("student_architecture"):
        model_args += ["--student_architecture", cfg["student_architecture"]]
    if st.get("intermediate"):
        model_args += ["--student_intermediate_size", str(st["intermediate"])]
    if st.get("prune_vocab"):
        model_args += ["--prune_student_vocab", str(st["prune_vocab"])]
    if st.get("prune_vocab_coverage"):
        model_args += ["--prune_vocab_coverage", str(st["prune_vocab_coverage"])]
    for key, flag in (("rope_qk_weight", "--rope_qk_weight"),
                      ("repr_weight", "--repr_weight"),
                      ("logit_kd_weight", "--logit_kd_weight"),
                      ("logit_kd_temperature", "--logit_kd_temperature"),
                      ("logit_kd_seq_chunk", "--logit_kd_seq_chunk")):
        if loss.get(key):
            model_args += [flag, str(loss[key])]
    if tr.get("teacher_dtype"):
        model_args += ["--teacher_dtype", tr["teacher_dtype"]]
    if tr.get("head_chunk_size") is not None:
        model_args += ["--head_chunk_size", str(tr["head_chunk_size"])]
    if tr.get("no_truncate_teacher"):
        model_args += ["--no_truncate_teacher"]
    if tr.get("compile", False):
        model_args += ["--compile_loss", "--compile_teacher"]
    if tr.get("compile_student"):
        model_args += ["--compile_student"]

    return data_args + train_args + model_args


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    cfg = yaml.safe_load(Path(sys.argv[1]).read_text())
    for override in sys.argv[2:]:  # dotted overrides: training.max_steps=2000
        key, _, val = override.partition("=")
        _set(cfg, key, val)

    argv = build_argv(cfg)
    import torch
    n_gpu = torch.cuda.device_count()
    if n_gpu > 1:
        cmd = ["uv", "run", "torchrun", f"--nproc_per_node={n_gpu}", "-m", "train.distillation", "--"]
    else:
        cmd = ["uv", "run", "python", "-m", "train.distillation", "--"]
    cmd += argv
    print("Running:", " ".join(cmd), flush=True)
    raise SystemExit(subprocess.call(cmd, cwd=REPO))


if __name__ == "__main__":
    main()
