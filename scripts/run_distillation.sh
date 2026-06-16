#!/usr/bin/env bash
#
# Generic launcher for MiniLMv2-style distillation (train.distillation).
#
# Replaces the legacy run_hplt76.sh / run_xlmr76.sh scripts. Everything is
# parameterized via environment variables (with sensible defaults) and a small
# set of CLI flags. Training is invoked through `uv run` (the project is
# managed with `uv sync`); multi-GPU runs use torchrun automatically.
#
# Configuration (env var, or CLI flag where listed):
#   TEACHER       (--teacher)       Teacher model name/path.
#                                   Default: FacebookAI/xlm-roberta-large
#   STUDENT_ARCH  (--student-arch)  Student architecture config (e.g.
#                                   jhu-clsp/mmBERT-base). Empty = BERT default.
#   DATASET       (--dataset)       HF dataset name. Default: uonlp/CulturaX
#   DATASET_CONFIG (--dataset-config) Dataset config/subset. Default: en
#   LANG_LABEL    (--lang)          Language label used in output paths.
#                                   Default: $DATASET_CONFIG. (Named LANG_LABEL
#                                   to avoid clobbering the locale LANG.)
#   HIDDEN        (--hidden)        Student hidden size.        Default: 768
#   LAYERS        (--layers)        Student layer count.        Default: 6
#   HEADS         (--heads)         Student attention heads.    Default: 12
#   INTERMEDIATE  (--intermediate)  Student FFN size (empty = architecture default).
#   L             (--L)             Teacher layer to distill from. Default: 12
#   A_R           (--ar)            Number of relation heads.   Default: 64
#   MAX_STEPS     (--max-steps)     Training steps.             Default: 200000
#   BATCH         (--batch)         GLOBAL batch size across GPUs. Default: 256
#   SEQ_LEN       (--seq-len)       Max sequence length.        Default: 512
#   PACK=1        (--pack)          Enable --pack (packed fixed-shape batches).
#   COMPILE=1     (--compile)       Enable --compile_loss and --compile_teacher.
#   PRUNE_VOCAB=N (--prune-vocab N) Prune student vocab to N ids (0 = off).
#   SEED                            Random seed.                Default: 21
#   LR                              Learning rate.              Default: 6e-4
#   SAVE_STEPS                      Checkpoint interval.        Default: 10000
#   WARMUP_STEPS                    LR warmup steps.            Default: 4000
#   OUTPUT_BASE                     Base output dir.            Default: ./models
#   NUM_WORKERS                     Dataloader workers.         Default: 16
#
# Example invocations:
#
#   (a) Original XLM-R-Large -> TiME-m recipe (English CulturaX):
#       TEACHER=FacebookAI/xlm-roberta-large L=12 A_R=64 \
#       HIDDEN=768 LAYERS=6 HEADS=12 \
#       bash scripts/run_distillation.sh
#
#   (b) mmBERT-base -> xs recipe (pruned vocab, pack + compile):
#       bash scripts/run_distillation.sh \
#         --teacher jhu-clsp/mmBERT-base \
#         --student-arch jhu-clsp/mmBERT-base \
#         --L 19 --ar 12 \
#         --hidden 384 --layers 6 --heads 6 \
#         --prune-vocab 64000 --pack --compile

set -euo pipefail
export TOKENIZERS_PARALLELISM=false

# --- Defaults (override via env or flags) ---
TEACHER="${TEACHER:-FacebookAI/xlm-roberta-large}"
STUDENT_ARCH="${STUDENT_ARCH:-}"
DATASET="${DATASET:-uonlp/CulturaX}"
DATASET_CONFIG="${DATASET_CONFIG:-en}"
LANG_LABEL="${LANG_LABEL:-}"
HIDDEN="${HIDDEN:-768}"
LAYERS="${LAYERS:-6}"
HEADS="${HEADS:-12}"
INTERMEDIATE="${INTERMEDIATE:-}"
L="${L:-12}"
A_R="${A_R:-64}"
MAX_STEPS="${MAX_STEPS:-200000}"
BATCH="${BATCH:-256}"
SEQ_LEN="${SEQ_LEN:-512}"
PACK="${PACK:-0}"
COMPILE="${COMPILE:-0}"
PRUNE_VOCAB="${PRUNE_VOCAB:-0}"
SEED="${SEED:-21}"
LR="${LR:-6e-4}"
SAVE_STEPS="${SAVE_STEPS:-10000}"
WARMUP_STEPS="${WARMUP_STEPS:-4000}"
OUTPUT_BASE="${OUTPUT_BASE:-./models}"
NUM_WORKERS="${NUM_WORKERS:-16}"
RELATIONS="${RELATIONS:-{(1,1):1,(2,2):1,(3,3):1}}"

usage() { grep '^#' "$0" | sed 's/^# \{0,1\}//'; }

# --- CLI flags (override env/defaults) ---
while [[ $# -gt 0 ]]; do
    case "$1" in
        --teacher)        TEACHER="$2"; shift 2 ;;
        --student-arch)   STUDENT_ARCH="$2"; shift 2 ;;
        --dataset)        DATASET="$2"; shift 2 ;;
        --dataset-config) DATASET_CONFIG="$2"; shift 2 ;;
        --lang)           LANG_LABEL="$2"; shift 2 ;;
        --hidden)         HIDDEN="$2"; shift 2 ;;
        --layers)         LAYERS="$2"; shift 2 ;;
        --heads)          HEADS="$2"; shift 2 ;;
        --intermediate)   INTERMEDIATE="$2"; shift 2 ;;
        --L)              L="$2"; shift 2 ;;
        --ar)             A_R="$2"; shift 2 ;;
        --max-steps)      MAX_STEPS="$2"; shift 2 ;;
        --batch)          BATCH="$2"; shift 2 ;;
        --seq-len)        SEQ_LEN="$2"; shift 2 ;;
        --pack)           PACK=1; shift ;;
        --compile)        COMPILE=1; shift ;;
        --prune-vocab)    PRUNE_VOCAB="$2"; shift 2 ;;
        -h|--help)        usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 1 ;;
    esac
done

LANG_LABEL="${LANG_LABEL:-$DATASET_CONFIG}"

# --- GPU detection and per-GPU batch derivation ---
GPU_COUNT=0
if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l | tr -d ' ')
fi
echo "Detected GPU_COUNT: $GPU_COUNT"

if [[ "$GPU_COUNT" -gt 0 ]]; then
    PER_GPU_BATCH=$((BATCH / GPU_COUNT))
    if [[ "$PER_GPU_BATCH" -lt 1 ]]; then
        echo "Error: BATCH ($BATCH) is too small for GPU_COUNT ($GPU_COUNT)." >&2
        exit 1
    fi
    if (( BATCH % GPU_COUNT != 0 )); then
        BATCH=$((GPU_COUNT * PER_GPU_BATCH))
        echo "Warning: adjusted global BATCH to $BATCH for even division across $GPU_COUNT GPUs"
    fi
else
    PER_GPU_BATCH=$BATCH
    echo "Warning: no GPUs detected; running with PER_GPU_BATCH = BATCH = $BATCH (CPU)."
fi

# --- Output / checkpoint directory ---
DATASET_PATH_PART="${DATASET//\//_}_${DATASET_CONFIG}"
TEACHER_PATH_PART="${TEACHER//\//_}"
CHECKPOINT_DIR="${OUTPUT_BASE}/${DATASET_PATH_PART}/minilm-H${HIDDEN}-L${LAYERS}-${LANG_LABEL}-${TEACHER_PATH_PART}"
mkdir -p "$CHECKPOINT_DIR"

# --- Checkpoint-resume detection ---
RESUME_ARGS=()
LATEST_CHECKPOINT=$(find "$CHECKPOINT_DIR" -maxdepth 1 -type d -name 'checkpoint-*' | sort -t- -k2 -n | tail -n1)
if [[ -n "$LATEST_CHECKPOINT" ]]; then
    echo "Resuming from latest checkpoint: $LATEST_CHECKPOINT"
    RESUME_ARGS=(--resume_from_checkpoint "$LATEST_CHECKPOINT")
elif [[ -f "$CHECKPOINT_DIR/pytorch_model.bin" || -f "$CHECKPOINT_DIR/model.safetensors" ]]; then
    echo "Found model files in $CHECKPOINT_DIR; letting Trainer locate a resume point."
    RESUME_ARGS=(--resume_from_checkpoint True)
else
    echo "No previous checkpoint in $CHECKPOINT_DIR; starting fresh."
fi

# --- Optional flags ---
DATA_EXTRA=()
MODEL_EXTRA=()
[[ "$PACK" == "1" ]] && DATA_EXTRA+=(--pack)
[[ "$COMPILE" == "1" ]] && MODEL_EXTRA+=(--compile_loss --compile_teacher)
[[ -n "$STUDENT_ARCH" ]] && MODEL_EXTRA+=(--student_architecture "$STUDENT_ARCH")
[[ -n "$INTERMEDIATE" ]] && MODEL_EXTRA+=(--student_intermediate_size "$INTERMEDIATE")
[[ "$PRUNE_VOCAB" != "0" ]] && MODEL_EXTRA+=(--prune_student_vocab "$PRUNE_VOCAB")

ARGS=(
    data_params
      --max_seq_len "$SEQ_LEN"
      --stream_local_files
      --dataset_name "$DATASET"
      --dataset_config_name "$DATASET_CONFIG"
      "${DATA_EXTRA[@]}"
    training_params
      --per_device_train_batch_size "$PER_GPU_BATCH"
      --learning_rate "$LR"
      --adam_epsilon 1e-6
      --adam_beta1 0.9
      --adam_beta2 0.999
      --weight_decay 0.01
      --max_steps "$MAX_STEPS"
      --save_strategy steps
      --save_steps "$SAVE_STEPS"
      --logging_strategy steps
      --logging_steps 100
      --warmup_steps "$WARMUP_STEPS"
      --gradient_accumulation_steps 1
      --bf16 true
      --dataloader_drop_last true
      --max_grad_norm 1.0
      --ddp_find_unused_parameters true
      --output_dir "$CHECKPOINT_DIR"
      --seed "$SEED"
      --dataloader_num_workers "$NUM_WORKERS"
      "${RESUME_ARGS[@]}"
    model_params
      --input_model_dir "$TEACHER"
      --student_hidden_size "$HIDDEN"
      --student_num_layers "$LAYERS"
      --student_attention_heads "$HEADS"
      --L "$L"
      --num_relation_heads "$A_R"
      --minilm_relations "$RELATIONS"
      "${MODEL_EXTRA[@]}"
)

if [[ "$GPU_COUNT" -gt 1 ]]; then
    CMD=(uv run torchrun --nproc_per_node="$GPU_COUNT" -m train.distillation --)
else
    CMD=(uv run python -m train.distillation --)
fi

echo "Executing: ${CMD[*]} ${ARGS[*]}"
"${CMD[@]}" "${ARGS[@]}"

# --- model_details.txt (consumed by evaluation/find_best_checkpoint) ---
# Key names below are a contract with read_model_details(); do not rename.
DETAILS="$CHECKPOINT_DIR/model_details.txt"
echo "Saving model details to $DETAILS"
{
    echo "Teacher_Model: $TEACHER"
    echo "Student_Architecture: ${STUDENT_ARCH:-bert}"
    echo "Student_Hidden_Size: $HIDDEN"
    echo "Student_Num_Layers: $LAYERS"
    echo "Student_Attention_Heads: $HEADS"
    echo "Teacher_Distillation_Layer (L): $L"
    echo "Num_Relation_Heads: $A_R"
    echo "Minilm_Relations: $RELATIONS"
    echo "Output Directory (contains checkpoints): $CHECKPOINT_DIR"
    echo "Seed: $SEED"
    echo "Full Batch Size (across all GPUs): $BATCH"
    echo "Per GPU Batch Size: $PER_GPU_BATCH"
    echo "GPU Count: $GPU_COUNT"
    echo "Max Seq Len: $SEQ_LEN"
    echo "Pruned Student Vocab: $PRUNE_VOCAB"
    echo "Final Command Args passed to Python script: ${ARGS[*]}"
} > "$DETAILS"
