# NLP Downstream Task Evaluation

This directory contains the scripts and utilities for evaluating the fine-tuned performance of our distilled language models on core NLP tasks. The evaluation pipeline is divided into two main components:

1.  **Universal Dependencies (UD):** For Part-of-Speech (POS) tagging, lemmatization, and dependency parsing (LAS).
2.  **Named Entity Recognition (NER):** For evaluating token classification on the WikiAnn dataset.
3.  **German Text Classification (textcls):** For evaluating sequence classification (sentence-level) on German news-topic (10kGNAD) and offensive-language (GermEval2018) benchmarks. These are single-text classification tasks comparable to the classification slice of **SuperGLEBer** (the German NLU benchmark used as the headline suite for German encoders such as ModernGBERT); they are the most directly relevant signal for distilling German classification students.

## Acknowledgement

The evaluation framework in this directory, particularly the Universal Dependencies pipeline (`ud/`), is heavily based on and adapted from the excellent evaluation suite developed for the **HPLT project**. We extend our sincere gratitude to the HPLT authors for making their robust and comprehensive evaluation code publicly available, which provided a strong foundation for our benchmarking.

Original HPLT evaluation resources can be found with their model releases. (https://github.com/hplt-project/HPLT-WP4)
## Directory Structure

-   **`/ner`**: Contains all scripts and utilities for running NER evaluation.
    -   `ner_eval.sh`: The main bash script to orchestrate the NER fine-tuning and evaluation process.
    -   `ner_bert.py`: The core Python script that handles model loading, training, and prediction using the `transformers` library.
    -   `ner_eval.py`: A helper script for calculating entity-level metrics.
    -   `tsa_utils.py`: Contains dataclasses and helper functions for the NER pipeline.
    -   `constants.py`: Holds language code mappings.

-   **`/ud`**: Contains all scripts and utilities for running the UD evaluation (POS, Lemmatization, Parsing).
    -   `run_ud_eval.sh`: The main bash script to run the complete UD pipeline for a given language and model.
    -   `train.py`: The primary Python script that trains the multi-task UD model.
    -   `model.py`: Defines the neural network architecture for the UD tasks.
    -   `lemma_rule.py`: Implements the rule-based lemmatization logic.
    -   `/ud_data/`: A placeholder directory where the user must place the downloaded Universal Dependencies treebank files (e.g., from `v2.15`).

-   **`/textcls`**: German sentence-level text-classification evaluation. Fine-tunes a
    classification head (`AutoModelForSequenceClassification`) on top of an arbitrary
    encoder checkpoint (e.g. a distilled German ModernBERT student) and reports a clean
    accuracy + macro-F1 number as JSON.
    -   `run_textcls_eval.sh`: Thin wrapper, mirrors `ner_eval.sh`.
    -   `german_textcls.py`: Core fine-tune + eval script.

## German Text Classification

`textcls/german_textcls.py` fine-tunes a sequence-classification head on a given
HuggingFace encoder checkpoint and reports **test/validation accuracy + macro-F1** as
JSON, written to `<output_dir>/textcls_results.json` (plus a one-line append to
`textcls_results.tsv`). It runs with the project's base environment only
(`transformers` + `torch` + `datasets`); accuracy and macro-F1 are computed in-script
with NumPy, so no `evaluate`/`scikit-learn` install is required.

### Datasets

| `--dataset`     | Hub id                      | Task                                         | Classes | Text col | Label col |
|-----------------|-----------------------------|----------------------------------------------|---------|----------|-----------|
| `10kgnad`       | `community-datasets/gnad10` | German news topic classification (10kGNAD)   | 9       | `text`   | `label`   |
| `germeval2018`  | `philschmid/germeval18`     | German tweet offensive-language (coarse)     | 2       | `text`   | `binary`  |

Notes:
-   **10kGNAD** label set: `Web, Panorama, International, Wirtschaft, Sport, Inland, Etat,
    Wissenschaft, Kultur` (a `ClassLabel` feature; the names are read straight from the
    dataset). Splits as shipped: train 9245 / test 1028.
-   **GermEval2018** is used here for the **coarse binary** task (`binary` column:
    `OFFENSE` / `OTHER`); the dataset also has a fine-grained `multi` column that this
    script does not use. Splits as shipped: train 5009 / test 3398.
-   Neither dataset ships a validation split, so the script carves one off the train split
    (`--validation_split`, default 0.1) for `load_best_model_at_end` (best by macro-F1).
-   The first attempted 10kGNAD id `community-datasets/10k_gnad` / `gnad10` did **not**
    resolve on the current Hub; `community-datasets/gnad10` is the working canonical id.

### Usage

Via the wrapper (`[dataset] [model_path] [output_dir]`):

```bash
cd evaluation/nlp_eval/textcls
./run_textcls_eval.sh 10kgnad      jhu-clsp/mmBERT-base
./run_textcls_eval.sh germeval2018 jhu-clsp/mmBERT-base
```

Or directly with `uv run`:

```bash
uv run python german_textcls.py \
    --model_name_or_path <hf-checkpoint-or-path> \
    --dataset 10kgnad \
    --output_dir ./results/german_textcls_10kgnad/<model>
```

Default training hyper-parameters: 4 epochs, lr 2e-5, batch 32, max_len 256 (a standard
HF `Trainer` fine-tune; designed for GPU). For a quick CPU smoke test:

```bash
uv run python german_textcls.py \
    --model_name_or_path hf-internal-testing/tiny-random-BertModel \
    --dataset germeval2018 --output_dir /tmp/smoke \
    --max_train_samples 32 --max_eval_samples 16 --max_steps 2 \
    --num_train_epochs 1 --max_seq_length 64
```

### Expected metric

The headline number is **test macro-F1** (with test accuracy alongside). For a
well-trained German encoder, 10kGNAD typically lands around the high-0.8s accuracy and
GermEval2018 binary around the mid-0.7s F1; a tiny distilled student should be compared
against its teacher under identical settings.

## How to Use

Instructions for running these evaluation scripts are provided in the main `README.md` at the root of the repository. Please refer to the **"How to Run" -> "Downstream Task Evaluation"** section for detailed commands.

### Prerequisites

Before running the UD evaluation, you must download the required Universal Dependencies treebanks and place them into the `ud/ud_data/` directory.

```
evaluation/nlp_eval/ud/
└── ud_data/
    └── ud-treebanks-v2.15/
        ├── UD_English-EWT/
        ├── UD_German-GSD/
        └── ...
```