#!/usr/bin/env bash
# Thin wrapper around ner_bert.py for NER (WikiAnn) fine-tuning + evaluation.
#
# Usage:
#   ./ner_eval.sh [dataset] <model_path> [output_dir]
#
#   dataset    WikiAnn split, e.g. wikiann/de   (default: wikiann/de)
#   model_path HF id or local path to an encoder checkpoint
#   output_dir where to write metrics JSON       (default: ./results/ner/<model>)
#
# This repo uses uv (see pyproject.toml; benchmark deps: `uv sync --extra eval`),
# so Python runs through `uv run` rather than a hard-coded interpreter path.
set -euo pipefail

# ner_bert.py imports sibling modules, so it must run from its own directory;
# resolve user-relative paths against the invoking cwd before changing into it.
invoker_pwd="$PWD"
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

dataset_name="${1:-wikiann/de}"
model_path="${2:?usage: ./ner_eval.sh [dataset] <model_path> [output_dir]}"
# absolutize a local checkpoint path; HF hub ids (not real paths) pass through
[ -e "$invoker_pwd/$model_path" ] && model_path="$(cd "$invoker_pwd/$(dirname "$model_path")" && pwd)/$(basename "$model_path")"
output_dir="${3:-./results/ner/${model_path//\//-}}"
case "$output_dir" in /*) ;; *) output_dir="$invoker_pwd/$output_dir" ;; esac

uv run python -u ner_bert.py \
    --model_name_or_path "$model_path" \
    --dataset_name "$dataset_name" \
    --output_dir "$output_dir" \
    --trust_remote_code True
