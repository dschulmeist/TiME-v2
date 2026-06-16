#!/usr/bin/env bash
# Universal Dependencies (POS / Lemma / LAS) fine-tuning + evaluation for a model.
#
# Usage:
#   ./run_ud_eval.sh <language> <model_path> [treebank_root] [epochs]
#
#   language       UD language code, e.g. de (mapped to a treebank via
#                  language_treebank_mapping_2_0.json)
#   model_path     HF id or local path to an encoder checkpoint
#   treebank_root  dir holding the UD treebanks (default: ./ud_data/ud-treebanks-v2.15)
#   epochs         training epochs (default: 30)
#
# This repo uses uv (see pyproject.toml; benchmark deps: `uv sync --extra eval`),
# so Python runs through `uv run` rather than a hard-coded interpreter path.
set -euo pipefail

# train.py imports sibling modules, so it must run from its own directory;
# resolve user-relative paths against the invoking cwd before changing into it.
invoker_pwd="$PWD"
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

language="${1:?usage: ./run_ud_eval.sh <language> <model_path> [treebank_root] [epochs]}"
model_path="${2:?usage: ./run_ud_eval.sh <language> <model_path> [treebank_root] [epochs]}"
treebank_root="${3:-./ud_data/ud-treebanks-v2.15}"
epochs="${4:-30}"

# absolutize a local checkpoint path; HF hub ids (not real paths) pass through
[ -e "$invoker_pwd/$model_path" ] && model_path="$(cd "$invoker_pwd/$(dirname "$model_path")" && pwd)/$(basename "$model_path")"
case "$treebank_root" in /*) ;; *) [ -e "$invoker_pwd/$treebank_root" ] && treebank_root="$invoker_pwd/$treebank_root" ;; esac

out_dir="$invoker_pwd/results/ud/${model_path//\//-}"
mkdir -p "${out_dir}/checkpoints"

uv run python -u train.py \
    --model hf --custom_model_path "$model_path" \
    --language "$language" \
    --treebank_path "$treebank_root" \
    --version 2_0 \
    --epochs "$epochs" \
    --batch_size 32 --lr 2e-5 --dropout 0.3 --min_count 3 \
    --results_path "${out_dir}/" \
    --checkpoints_path "${out_dir}/checkpoints/"
