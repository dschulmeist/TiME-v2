#!/bin/bash
# 2x2 distillation matrix + shared evaluation.
#   teacher in {ModernGBERT-1B, mmBERT-base} x loss in {relation, relation+logit-KD}
# Trains a small German ModernBERT student per cell, then evaluates all four on
# the same German classification benchmarks and prints a cost+quality table.
#
#   bash experiments/run_all.sh                # full 15k-step runs
#   STEPS=300 bash experiments/run_all.sh       # quick end-to-end dry run
#   PARALLEL=1 bash experiments/run_all.sh      # force one-run-per-GPU (auto when >=4 GPUs)
set -euo pipefail
cd "$(dirname "$0")/.."
# fast tokenizers tokenize single-threaded inside each worker process (avoids
# Rust-thread x process oversubscription); BLAS likewise pinned low per process.
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
STEPS="${STEPS:-15000}"

RUNS=(run1_moderngbert_relation run2_moderngbert_logit run3_mmbert_relation run4_mmbert_logit)

NGPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l || echo 0)
PARALLEL="${PARALLEL:-$([ "$NGPU" -ge 4 ] && echo 1 || echo 0)}"

# Saturate the node's CPUs with dataloader workers: divide cores across the
# concurrently-running training processes (one per GPU when parallel), leaving
# a small margin for the main processes, so the GPUs stay fed during streaming.
NCPU=$(nproc --all 2>/dev/null || nproc 2>/dev/null || echo 8)
NPROC=$([ "$PARALLEL" = "1" ] && echo "${#RUNS[@]}" || echo 1)
# streaming workers each hold a shuffle buffer (RAM-heavy on large corpora), so
# cap conservatively: a handful already saturates one GPU, and too many OOM the
# host. The cap matters most in the single-GPU sequential case.
WORKERS="${WORKERS:-$(( (NCPU - NPROC) / NPROC ))}"
[ "$WORKERS" -lt 2 ] && WORKERS=2
[ "$WORKERS" -gt 8 ] && WORKERS=8
echo "Detected $NGPU GPU(s), $NCPU CPU(s); parallel=$PARALLEL; workers/run=$WORKERS; max_steps=$STEPS"

# frequent checkpoints: cheap (small students) and minimise lost progress if a
# run is interrupted and resumed. Caps to the step count for short dry runs.
SAVE=$([ "$STEPS" -lt 1000 ] && echo "$STEPS" || echo 1000)

train_one() {  # $1=run index, $2=run name
  local gpu=$1 name=$2
  CUDA_VISIBLE_DEVICES="$gpu" uv run python scripts/run_experiment.py \
    "experiments/$name.yaml" "training.max_steps=$STEPS" "training.save_steps=$SAVE" \
    "training.num_workers=$WORKERS" > "models/$name.train.log" 2>&1
}

echo "=== TRAIN ==="
mkdir -p models
if [ "$PARALLEL" = "1" ]; then
  pids=()
  for i in "${!RUNS[@]}"; do
    echo "--- launching ${RUNS[$i]} on GPU $i ---"
    train_one "$i" "${RUNS[$i]}" & pids+=($!)
  done
  fail=0
  for p in "${pids[@]}"; do wait "$p" || fail=1; done
  [ "$fail" = 1 ] && echo "WARNING: a training run failed; see models/*.train.log"
else
  for name in "${RUNS[@]}"; do
    echo "--- training $name ---"
    train_one 0 "$name" || echo "WARNING: training $name failed (continuing to next + eval)"
  done
fi

echo "=== EXPORT + EVAL each student on German benchmarks ==="
mkdir -p results  # eval logs redirect here; the dir must exist before the redirect
eval_one() {  # $1=gpu, $2=run name
  local gpu=$1 name=$2
  local ckpt
  ckpt=$(ls -d "models/$name"/student/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
  [ -z "$ckpt" ] && { echo "no checkpoint for $name, skipping"; return; }
  for ds in 10kgnad germeval2018; do
    CUDA_VISIBLE_DEVICES="$gpu" uv run python evaluation/nlp_eval/textcls/german_textcls.py \
      --model_name_or_path "$ckpt" --dataset "$ds" --output_dir "results/$name/$ds" \
      > "results/$name.$ds.eval.log" 2>&1 || echo "eval $name/$ds failed"
  done
}
if [ "$PARALLEL" = "1" ]; then
  pids=()
  for i in "${!RUNS[@]}"; do eval_one "$i" "${RUNS[$i]}" & pids+=($!); done
  for p in "${pids[@]}"; do wait "$p" || true; done
else
  for name in "${RUNS[@]}"; do eval_one 0 "$name"; done
fi

echo "=== SUMMARY: efficiency + accuracy ==="
uv run python scripts/summarize_experiments.py --runs "${RUNS[@]}" || true
echo "=== DONE. Per-run: models/<run>/efficiency.json + results/<run>/<ds>/. Joined: results/summary.csv ==="
