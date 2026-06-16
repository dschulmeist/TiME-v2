#!/usr/bin/env bash
# Run the German TiME benchmark suite (text-cls + NER + UD) on one or more models.
# Each argument is  name=model  where model is a local checkpoint dir or an HF id.
#
# Models run concurrently (JOBS, default 3) to fill the GPU - small students
# leave most of an 80GB card idle one-at-a-time. Within a model the tasks run
# sequentially. Batch sizes are the eval defaults (paper-faithful); parallelism
# comes from running models side by side, not from changing batch size.
#
# Setup once:   uv sync --extra eval
# Run:
#   bash evaluation/run_german_benchmarks.sh \
#       german_moderngbert=./models/german_moderngbert/student/checkpoint-6000 \
#       moderngbert_teacher=LSX-UniWue/ModernGBERT_1B \
#       mmbert_teacher=jhu-clsp/mmBERT-base
#
# Knobs (env, defaults shown):
#   JOBS=3                concurrent models (lower to 1 for a very large teacher)
#   TASKS=textcls,ner,ud  which tasks to run
#   TEXTCLS_EPOCHS=4  NER_EPOCHS=5  UD_EPOCHS=30   fine-tuning epochs per task
#   MAX_TRAIN= MAX_EVAL=  cap train/eval samples (text-cls + NER); empty = full
# For a fast pipeline check (seconds per task, throwaway quality):
#   TEXTCLS_EPOCHS=1 NER_EPOCHS=1 UD_EPOCHS=1 MAX_TRAIN=256 MAX_EVAL=128 bash ...
# A large teacher can OOM an 80GB card at JOBS>1; run such models with JOBS=1.
#
# Results land in  results/<name>/{textcls_*,ner,ud}/  - summarise with:
#   uv run python evaluation/summarize_german_benchmarks.py <name> ...
# Exit status is non-zero if any model/task failed (the failures are listed).
set -uo pipefail
cd "$(dirname "$0")/.."
REPO=$(pwd)
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
JOBS="${JOBS:-3}"
TASKS="${TASKS:-textcls,ner,ud}"
TEXTCLS_EPOCHS="${TEXTCLS_EPOCHS:-4}"
NER_EPOCHS="${NER_EPOCHS:-5}"
UD_EPOCHS="${UD_EPOCHS:-30}"
MAX_TRAIN="${MAX_TRAIN:-}"
MAX_EVAL="${MAX_EVAL:-}"

[ $# -eq 0 ] && { sed -n '2,33p' "$0" | sed -e 's/^# //' -e 's/^#//'; exit 1; }

echo "Benchmarking $# model(s); JOBS=$JOBS, TASKS=$TASKS, UD_EPOCHS=$UD_EPOCHS."
[ "$JOBS" -gt 1 ] && echo "  (set JOBS=1 if any model is a very large teacher - concurrent fine-tunes can OOM.)"

# UD treebank: fetch German-GSD (UD v2.15) into the path the mapping expects.
# Check for the actual train .conllu (not just the dir) so a partial/failed
# clone is detected and retried rather than silently breaking UD downstream.
TBROOT="$REPO/evaluation/nlp_eval/ud/ud_data/ud-treebanks-v2.15"
if [[ "$TASKS" == *ud* ]] && ! ls "$TBROOT/UD_German-GSD"/*train*.conllu >/dev/null 2>&1; then
  echo "Fetching UD_German-GSD (v2.15)..."
  rm -rf "$TBROOT/UD_German-GSD"          # clear any partial clone
  mkdir -p "$TBROOT"
  git clone --depth 1 --branch r2.15 https://github.com/UniversalDependencies/UD_German-GSD.git \
       "$TBROOT/UD_German-GSD" \
   || git clone --depth 1 https://github.com/UniversalDependencies/UD_German-GSD.git "$TBROOT/UD_German-GSD" \
   || { echo "ERROR: failed to clone UD_German-GSD treebank"; exit 1; }
  ls "$TBROOT/UD_German-GSD"/*train*.conllu >/dev/null 2>&1 \
   || { echo "ERROR: UD_German-GSD clone has no train .conllu"; exit 1; }
fi

# Per model: run the requested tasks; write a .status file (ok / FAILED:<tasks>)
# so the parallel launcher can report failures even though jobs run detached.
bench_one() {
  local name=$1 model=$2 failed=""
  # absolutize a local checkpoint path so it survives the cd into the NER/UD
  # script directories below; HF hub ids (not real paths) are left unchanged
  [ -e "$model" ] && model="$(cd "$(dirname "$model")" && pwd)/$(basename "$model")"
  mkdir -p "$REPO/results/$name"
  local caps=()
  [ -n "$MAX_TRAIN" ] && caps+=(--max_train_samples "$MAX_TRAIN")
  [ -n "$MAX_EVAL" ] && caps+=(--max_eval_samples "$MAX_EVAL")
  if [[ "$TASKS" == *textcls* ]]; then
    for ds in 10kgnad germeval2018; do
      uv run --no-sync python evaluation/nlp_eval/textcls/german_textcls.py \
        --model_name_or_path "$model" --dataset "$ds" --num_train_epochs "$TEXTCLS_EPOCHS" "${caps[@]}" \
        --output_dir "$REPO/results/$name/textcls_$ds" \
        > "$REPO/results/$name/textcls_$ds.log" 2>&1 || failed="$failed textcls/$ds"
    done
  fi
  if [[ "$TASKS" == *ner* ]]; then
    ( cd evaluation/nlp_eval/ner && uv run --no-sync python -u ner_bert.py \
        --model_name_or_path "$model" --dataset_name wikiann/de --num_train_epochs "$NER_EPOCHS" "${caps[@]}" \
        --output_dir "$REPO/results/$name/ner" --trust_remote_code True \
        > "$REPO/results/$name/ner.log" 2>&1 ) || failed="$failed ner"
  fi
  if [[ "$TASKS" == *ud* ]]; then
    mkdir -p "$REPO/results/$name/ud/checkpoints"
    ( cd evaluation/nlp_eval/ud && uv run --no-sync python -u train.py \
        --model hf --custom_model_path "$model" --language de \
        --treebank_path "$TBROOT" --version 2_0 --epochs "$UD_EPOCHS" \
        --batch_size 32 --lr 2e-5 --dropout 0.3 --min_count 3 \
        --results_path "$REPO/results/$name/ud/" \
        --checkpoints_path "$REPO/results/$name/ud/checkpoints/" \
        > "$REPO/results/$name/ud.log" 2>&1 ) || failed="$failed ud"
  fi
  if [ -n "$failed" ]; then
    echo "FAILED:$failed" > "$REPO/results/$name/.status"
    echo "FAILED $name - $failed"
  else
    echo "ok" > "$REPO/results/$name/.status"
    echo "done: $name"
  fi
}

for spec in "$@"; do
  name="${spec%%=*}"; model="${spec#*=}"
  rm -f "$REPO/results/$name/.status"
  echo "--- queued $name ($model) ---"
  bench_one "$name" "$model" &
  while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do wait -n; done
done
wait

# Report: collect any model whose .status is not "ok".
failed=()
for spec in "$@"; do
  name="${spec%%=*}"
  [ "$(cat "$REPO/results/$name/.status" 2>/dev/null)" = "ok" ] || failed+=("$name")
done
if [ "${#failed[@]}" -gt 0 ]; then
  echo "BENCHMARKS FINISHED WITH FAILURES: ${failed[*]} (see results/<name>/.status and *.log)"
  exit 1
fi
echo "ALL_BENCHMARKS_DONE - results in results/<name>/"
