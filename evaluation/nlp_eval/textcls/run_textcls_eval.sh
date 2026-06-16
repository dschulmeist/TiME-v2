#!/usr/bin/env bash
# Thin wrapper around german_textcls.py, mirroring ner_eval.sh.
#
# Usage:
#   ./run_textcls_eval.sh [dataset] [model_path] [output_dir]
#
#   dataset    one of: 10kgnad | germeval2018   (default: 10kgnad)
#   model_path HF id or local path to an encoder checkpoint
#              (default: FacebookAI/roberta-base)
#   output_dir where to write checkpoints + metrics JSON
#
# This repo uses uv (pyproject.toml: transformers 5.11, torch 2.12), so we run
# Python through `uv run` rather than a hard-coded interpreter path.
set -euo pipefail

working_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$working_dir"

python_script="german_textcls.py"

dataset_name="10kgnad"
model_path="FacebookAI/roberta-base"

first_arg="${1:-}"
second_arg="${2:-}"
third_arg="${3:-}"

if [ -n "$first_arg" ]; then
    dataset_name=$first_arg
fi
if [ -n "$second_arg" ]; then
    model_path=$second_arg
fi

model_path_sanitized=${model_path//\//-} # Replaces all '/' with '-'
output_dir="./results/german_textcls_${dataset_name}/${model_path_sanitized}"
if [ -n "$third_arg" ]; then
    output_dir=$third_arg
fi

set +e
uv run python -u "$python_script" \
    --model_name_or_path "$model_path" \
    --dataset "$dataset_name" \
    --output_dir "$output_dir" \
    --trust_remote_code
exit_code=$?
set -e

if [ $exit_code -eq 0 ]; then
    echo "Success!"
else
    echo "Error with Exit-Code $exit_code" >&2
    exit $exit_code
fi
