# Tiny Monolingual Encoders

This repository contains the official code and resources for the paper **"Tiny Language Models for NLP Pipelines"**. Our work demonstrates how to train small, efficient, and high-performing monolingual language models for common NLP tasks using knowledge distillation.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Setup and Installation](#setup-and-installation)
- [Running the Tests](#running-the-tests)
- [How to Run](#how-to-run)
  - [1. Distillation (Training)](#1-distillation-training)
  - [2. Finding the Best Checkpoint](#2-finding-the-best-checkpoint)
  - [3. Downstream Task Evaluation (UD & NER)](#3-downstream-task-evaluation-ud--ner)
- [Results](#results)
- [Citation](#citation)

## Overview

Large language models, while powerful, are often too slow and computationally expensive for real-time applications or large-scale data processing. This project addresses this gap by providing a complete pipeline to distill large teacher models (like XLM-RoBERTa and HPLT) into compact, purpose-built "tiny" student models.

The key components of this repository are:
- **Distillation Training:** Scripts to perform knowledge distillation using the MiniLMv2 methodology.
- **Checkpoint Selection:** A robust script to evaluate all saved checkpoints against a validation set to find the optimal one.
- **Downstream Evaluation:** A comprehensive evaluation suite for core NLP tasks, including Part-of-Speech (POS) tagging, lemmatization, dependency parsing (LAS), and Named Entity Recognition (NER).

## Project Structure

The repository is organized into two main parts: `train/` for model distillation and `evaluation/` for assessing model performance.

```
TiME/
├── train/                           # distillation core
│   ├── distillation.py              # training entry point
│   ├── distiller.py                 # MiniLMDistiller (one distiller for all teachers)
│   ├── losses.py                    # relation-KL, hidden-state, logit-KD
│   ├── adapters.py                  # per-architecture Q/K/V extraction + registry
│   ├── data_pipeline.py             # streaming + tokenization
│   ├── packing.py                   # sequence packing
│   └── vocab_pruning.py             # student vocabulary pruning
├── experiments/                     # ready-made configs + study matrix
│   ├── german_moderngbert.yaml      # German recipe (ModernGBERT-1B teacher)
│   ├── german_mmbert.yaml           # German recipe (mmBERT-base teacher)
│   └── run_all.sh                   # 2x2 study matrix (run1-4)
├── scripts/
│   ├── run_experiment.py            # run a YAML experiment config
│   ├── run_distillation.sh          # generic env-var / CLI training launcher
│   └── ...                          # export, checkpoint selection, profiling
├── evaluation/
│   ├── run_german_benchmarks.sh     # text-cls + NER + UD on any model(s)
│   ├── summarize_german_benchmarks.py
│   ├── nlp_eval/{ner,ud,textcls}/   # per-task fine-tune + eval
│   ├── energy_benchmark/            # throughput + GPU energy
│   ├── find_best_checkpoint/        # pick the best checkpoint by validation loss
│   └── spacy_evaluation/            # latency / throughput vs spaCy
├── tests/                           # pytest suite
├── pyproject.toml                   # dependencies (uv); 'eval' extra for benchmarks
└── README.md
```

Architecture notes:

- All distillation losses live in `train/losses.py`; there is exactly one
  implementation of the masked relation-KL, chunked over relation heads to
  bound peak GPU memory.
- Teacher architectures are supported via `QKVRecorder` subclasses in
  `train/adapters.py`, registered by `config.model_type`. Unknown
  architectures fail loudly instead of being silently treated as BERT.
- The teacher runs frozen in bf16 (on CUDA) and is truncated at the
  distillation layer `L` by default - layers above `L` contribute nothing.

## Setup and Installation

The project uses [uv](https://docs.astral.sh/uv/) for dependency management.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/TiME.git
    cd TiME
    ```

2.  **Install the dependencies (creates `.venv` automatically):**
    ```bash
    uv sync
    ```
    Core dependencies are declared in `pyproject.toml` and locked in `uv.lock`.
    The downstream-evaluation extras (seqeval, conllu, ufal.chu-liu-edmonds,
    etc.) are the `eval` optional group - install with `uv sync --extra eval`.

3.  **(Optional) Environment Variables:**
    If you need to access private models from the Hugging Face Hub, create a `.env` file in the root directory and add your token:
    ```
    HF_TOKEN=your_hugging_face_token_here
    ```

## Running the Tests

The distillation core (loss, adapters, distiller) is covered by a fast,
CPU-only test suite, including property tests for pad invariance and
hook-vs-manual Q/K/V equivalence:

```bash
uv run pytest
```

## How to Run

The project workflow consists of three main stages: training the student models, finding the best checkpoint from the training run, and finally, evaluating that checkpoint on downstream NLP tasks.

### 1. Distillation (Training)

Training is launched via `scripts/run_distillation.sh`, a generic launcher
around `train/distillation.py`. It is configured through environment
variables (each with a sensible default) and/or a small set of CLI flags - 
see the header comment of the script for the full list. It detects the GPU
count, derives the per-GPU batch size from the global `BATCH`, uses
`torchrun` for multi-GPU runs, resumes automatically from the latest
checkpoint, and writes the `model_details.txt` metadata required by the
checkpoint selector. Training runs through `uv run`, so a prior `uv sync`
is all the setup needed.

-   **Original XLM-R-Large -> TiME-m recipe (English CulturaX):**
    ```bash
    TEACHER=FacebookAI/xlm-roberta-large L=12 A_R=64 \
    HIDDEN=768 LAYERS=6 HEADS=12 \
    bash scripts/run_distillation.sh
    ```

-   **mmBERT-base -> xs recipe (pruned vocab, pack + compile):**
    ```bash
    bash scripts/run_distillation.sh \
      --teacher jhu-clsp/mmBERT-base \
      --student-arch jhu-clsp/mmBERT-base \
      --L 19 --ar 12 \
      --hidden 384 --layers 6 --heads 6 \
      --prune-vocab 64000 --pack --compile
    ```

-   **German recipes (config-driven):** two ready-made German configs run via
    `scripts/run_experiment.py` - `german_moderngbert.yaml` (ModernGBERT-1B
    teacher) and `german_mmbert.yaml` (multilingual mmBERT-base teacher). Any
    field is overridable on the CLI.
    ```bash
    uv run python scripts/run_experiment.py experiments/german_moderngbert.yaml
    uv run python scripts/run_experiment.py experiments/german_mmbert.yaml
    ```

Checkpoints and logs are written under `models/` (override with
`OUTPUT_BASE`). Toggles: `--pack`/`PACK=1` enables packed fixed-shape
batches, `--compile`/`COMPILE=1` enables `--compile_loss` and
`--compile_teacher`, and `--prune-vocab N`/`PRUNE_VOCAB=N` prunes the
student vocabulary.

#### Efficiency and architecture flags

`train.distillation` supports (see `--help` of each parser section for details):

- `data_params --pack` - pack documents into fixed-length rows with
  block-diagonal attention (near-zero padding; ~2x throughput together with
  `--compile_loss`). Supported for BERT/RoBERTa/XLM-R and ModernBERT models.
- `model_params --compile_loss` - torch.compile the relation loss (best with
  `--pack`, which fixes tensor shapes). torch.compile needs a working C toolchain
  for Triton; on a bare image install it first (`apt-get install build-essential
  python3-dev`).
- `model_params --compile_student` - torch.compile the student (small gain).
- `model_params --student_architecture <name>` - student model family, e.g.
  `jhu-clsp/mmBERT-base` for a ModernBERT-style student (default: BERT).
- `model_params --prune_student_vocab N` - shrink the student's embedding
  table to the N most frequent token ids of the training corpus; the teacher
  keeps the full vocabulary. The kept ids are saved as `vocab_map.json`.
- `model_params --teacher_dtype {auto,bf16,fp16,fp32}`, `--head_chunk_size N`,
  `--no_truncate_teacher` - memory/precision controls; defaults are the fast,
  validated settings (bf16 teacher, truncated at L, chunked loss).

ModernBERT/mmBERT teachers: pick a global-attention layer for `--L`
(mmBERT-base: 1, 4, 7, 10, 13, 16, 19, 22; upper layers recommended, e.g. 19).
The trainer pins `attn_implementation="sdpa"` for ModernBERT automatically.

### 2. Finding the Best Checkpoint

After training, multiple checkpoints are saved. The `find_best_checkpoint.py` script evaluates each of these checkpoints against an unseen validation set to identify the one with the lowest distillation loss, indicating the best generalization. It reads each variant's training configuration from its `model_details.txt` and refuses to score a checkpoint whose configuration is incomplete (scoring under guessed hyperparameters would produce non-comparable losses).

-   **Run the script:**
    - Verify the `BASE_MODEL_DIR` path inside the script points to your training output directory.
    - Execute the script:
        ```bash
        uv run python evaluation/find_best_checkpoint/find_best_checkpoint.py --results_csv evaluation/find_best_checkpoint/my_results.csv
        ```
This will produce a CSV file (`my_results.csv`) ranking all checkpoints. You can then use the `checkpoint_path` of the top-ranked model for the final evaluation.

### 3. Downstream Task Evaluation (UD & NER)

We provide evaluation pipelines for Universal Dependencies tasks (POS, Lemma, LAS) and Named Entity Recognition (NER).

#### German benchmark suite (one command)

For German models, `evaluation/run_german_benchmarks.sh` runs the whole suite - 
text classification (10kGNAD, GermEval2018), NER (WikiANN-de) and UD
(German-GSD: POS/Lemma/LAS) - on any number of models. It auto-fetches the
German-GSD treebank and runs models **concurrently to fill the GPU** (small
students leave most of an 80GB card idle one-at-a-time); batch sizes stay at the
eval defaults so numbers remain paper-faithful.

```bash
uv sync --extra eval        # one-time: installs the benchmark deps
bash evaluation/run_german_benchmarks.sh \
    german_moderngbert=./models/german_moderngbert/student/checkpoint-6000 \
    moderngbert_teacher=LSX-UniWue/ModernGBERT_1B \
    mmbert_teacher=jhu-clsp/mmBERT-base
uv run python evaluation/summarize_german_benchmarks.py   # -> results/german_summary.csv
```

Knobs: `JOBS` (concurrent models, default 3 - lower it if including a very large
teacher), `UD_EPOCHS` (default 30), `TASKS` (default `textcls,ner,ud`). The
per-task scripts below can also be run individually.

#### Universal Dependencies (UD) Evaluation

The `run_ud_eval.sh` script handles the fine-tuning and evaluation on UD tasks.

-   **Prerequisites:**
    - Download the required UD treebanks (e.g., v2.15) and place them in the `evaluation/nlp_eval/ud/ud_data/` directory.

-   **Run the evaluation:**
    The script accepts the language code and a model (HF id or local checkpoint).
    ```bash
    # Usage: ./run_ud_eval.sh <language> <model> [treebank_root] [epochs]
    bash evaluation/nlp_eval/ud/run_ud_eval.sh de ./models/german_moderngbert/student/checkpoint-6000
    ```
    Results are written as JSONL files under `results/ud/<model>/`.

#### Named Entity Recognition (NER) Evaluation

The `ner_eval.sh` script handles evaluation on the WikiAnn dataset for NER.

-   **Run the evaluation:**
    The script takes the WikiAnn split and a model (HF id or local checkpoint).
    ```bash
    # Usage: ./ner_eval.sh [dataset] <model> [output_dir]
    bash evaluation/nlp_eval/ner/ner_eval.sh wikiann/de ./models/german_moderngbert/student/checkpoint-6000
    ```

## Results (small German modernBERT tests)

A small German study used this pipeline to distill compact 6-layer ModernBERT
students from two teachers (ModernGBERT-1B and multilingual mmBERT-base) with
MiniLMv2 relation distillation, optionally plus output-logit KD. Each student was
evaluated against its teacher on text classification (10kGNAD accuracy,
GermEval2018 macro-F1), NER (WikiAnn-de span-F1), and UD parsing (German-GSD LAS).

> **These are smoke-test-scale numbers, not benchmarked results.** Students were
> trained for only about 6k steps; a proper run is on the order of 100k to 400k
> steps. Single seed, single student size, one run per cell. Read every number
> below as directional only.

| model | params | 10kGNAD | GermEval F1 | NER F1 | UD LAS |
|---|---|---|---|---|---|
| ModernGBERT-1B (teacher) | ~1 B | 0.905 | 0.750 | 0.862 | 86.4 |
| student, relation | 19.5 M | 0.856 | 0.700 | 0.819 | 79.2 |
| student, relation + logit-KD | 19.7 M | 0.845 | 0.717 | 0.803 | 78.3 |
| mmBERT-base (teacher) | 307 M | 0.902 | 0.751 | 0.884 | 85.9 |
| student, relation | 105.8 M | 0.859 | 0.722 | 0.829 | 79.5 |
| student, relation + logit-KD | 106.2 M | 0.865 | 0.702 | 0.808 | 78.3 |

Even at this tiny budget the compact students recover most of the teacher quality
on tagging and classification, with the largest gaps on the harder structured
tasks (NER, dependency parsing). Relation-only distillation was competitive with
adding logit-KD while training markedly faster.

### Reproducing these runs

Train the four students (the 2x2 matrix), 6k steps each:

```bash
for cfg in run1_moderngbert_relation run2_moderngbert_logit \
           run3_mmbert_relation run4_mmbert_logit; do
  uv run python scripts/run_experiment.py experiments/$cfg.yaml \
    training.max_steps=6000 training.save_steps=1000
done
```

Benchmark every student and both teachers on the full suite (text-cls + NER + UD):

```bash
uv sync --extra eval
bash evaluation/run_german_benchmarks.sh \
  run1_moderngbert_relation=./models/run1_moderngbert_relation/student/checkpoint-6000 \
  run2_moderngbert_logit=./models/run2_moderngbert_logit/student/checkpoint-6000 \
  run3_mmbert_relation=./models/run3_mmbert_relation/student/checkpoint-6000 \
  run4_mmbert_logit=./models/run4_mmbert_logit/student/checkpoint-6000 \
  teacher_moderngbert=LSX-UniWue/ModernGBERT_1B \
  teacher_mmbert=jhu-clsp/mmBERT-base
uv run python evaluation/summarize_german_benchmarks.py   # -> results/german_summary.csv
```

The two `german_*.yaml` configs above are the relation-only recipes (run1 and
run3) packaged for reuse; the matrix also includes their logit-KD variants.
