"""
German text-classification fine-tune + eval script.

Mirrors the style of evaluation/nlp_eval/ner/ner_bert.py and applies the same
transformers-5 compatibility fixes:
  * no `overwrite_output_dir` arg on TrainingArguments
  * `label2id` / `id2label` written as plain dict[str, int] / dict[int, str]
  * `Trainer(processing_class=...)` instead of the removed `tokenizer=` kwarg

It fine-tunes a sequence-classification head (AutoModelForSequenceClassification)
on top of an arbitrary AutoModel encoder checkpoint (e.g. a distilled German
ModernBERT student) and reports test/validation accuracy + macro-F1 as JSON.

Supported datasets (verified to load with the current `datasets` library):
  * 10kgnad      -> community-datasets/gnad10
                    German news topic classification, 9 classes.
                    columns: text (str), label (ClassLabel; 9 named topics)
                    splits: train 9245 / test 1028  (no validation -> carved from train)
  * germeval2018 -> philschmid/germeval18
                    German tweet coarse offensive-language classification.
                    columns: text (str), binary (str: OFFENSE / OTHER), multi (str, unused)
                    splits: train 5009 / test 3398  (no validation -> carved from train)

Usage:
    uv run python german_textcls.py \
        --model_name_or_path <hf-checkpoint-or-path> \
        --dataset {10kgnad,germeval2018} \
        --output_dir ./results/...

Designed for GPU but CPU-runnable for a tiny smoke test via:
    --max_train_samples 32 --max_eval_samples 16 --num_train_epochs 1
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import transformers
from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_DISABLED"] = "true"

import sys as _sys
import pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))
from modernbert_compat import patch_hf_validator
patch_hf_validator()

print("Numpy:", np.version.version)
print("PyTorch:", torch.__version__)
print("Transformers:", transformers.__version__)

# --------------------------------------------------------------------------- #
# Dataset registry. Each entry knows its Hub id, text column, label column and
# (optionally) an explicit list of label names. When `label_names` is None the
# names are derived from the dataset's ClassLabel feature or from the observed
# string values.
# --------------------------------------------------------------------------- #
DATASETS = {
    "10kgnad": {
        "hub_id": "community-datasets/gnad10",
        "config": None,
        "text_column": "text",
        "label_column": "label",
        "label_names": None,  # ClassLabel with 9 named topics
        "description": "10kGNAD German news topic classification (9 classes)",
    },
    "germeval2018": {
        "hub_id": "philschmid/germeval18",
        "config": None,
        "text_column": "text",
        # `binary` = coarse OFFENSE/OTHER. (`multi` is the fine-grained variant.)
        "label_column": "binary",
        "label_names": None,  # plain string labels OFFENSE / OTHER
        "description": "GermEval2018 German tweet offensive-language (binary: OFFENSE/OTHER)",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune + evaluate a German text-classification head on a "
                    "HuggingFace AutoModel encoder checkpoint."
    )
    parser.add_argument("--model_name_or_path", required=True,
                        help="HF model id or local path to an encoder checkpoint.")
    parser.add_argument("--dataset", required=True, choices=list(DATASETS.keys()),
                        help="Which German classification benchmark to run.")
    parser.add_argument("--output_dir", required=True,
                        help="Where to write checkpoints + the metrics JSON.")
    parser.add_argument("--max_seq_length", type=int, default=256)
    parser.add_argument("--num_train_epochs", type=float, default=4.0)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--per_device_train_batch_size", type=int, default=32)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--trust_remote_code", action="store_true", default=False)
    # Knobs used by the CPU smoke test.
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=-1)
    # Fraction of the train split to hold out as validation when the dataset
    # ships no validation split.
    parser.add_argument("--validation_split", type=float, default=0.1)
    return parser.parse_args()


def build_label_mappings(dsd, label_column, explicit_names):
    """Return (label_names, label_to_id, is_class_label).

    Handles both integer-valued (ClassLabel / int64) and string-valued labels.
    """
    feature = dsd["train"].features[label_column]
    if explicit_names is not None:
        label_names = list(explicit_names)
        return label_names, {str(n): i for i, n in enumerate(label_names)}, False

    # ClassLabel: names live on the feature.
    if hasattr(feature, "names") and feature.names:
        label_names = list(feature.names)
        return label_names, {n: i for i, n in enumerate(label_names)}, True

    # Otherwise inspect the values across all splits.
    values = set()
    for split in dsd:
        values.update(dsd[split][label_column])
    if all(isinstance(v, (int, np.integer)) for v in values):
        label_names = [str(v) for v in sorted(values)]
        return label_names, {n: i for i, n in enumerate(label_names)}, True
    # String labels (e.g. GermEval OFFENSE/OTHER): sort for a stable mapping.
    label_names = sorted(str(v) for v in values)
    return label_names, {n: i for i, n in enumerate(label_names)}, False


def main():
    args = parse_args()
    cfg = DATASETS[args.dataset]
    text_column = cfg["text_column"]
    label_column = cfg["label_column"]

    transformers.set_seed(args.seed)

    print(f"Loading dataset '{cfg['hub_id']}' (config={cfg['config']}) -> {cfg['description']}")
    if cfg["config"]:
        dsd = load_dataset(cfg["hub_id"], cfg["config"])
    else:
        dsd = load_dataset(cfg["hub_id"])
    transformers.logging.set_verbosity_warning()

    # These datasets ship train + test only; carve a validation split off train.
    if "validation" not in dsd:
        split = dsd["train"].train_test_split(
            test_size=args.validation_split, seed=args.seed
        )
        dsd["train"] = split["train"]
        dsd["validation"] = split["test"]
    print("Splits:", {k: len(v) for k, v in dsd.items()})

    label_names, label_to_id, label_is_int = build_label_mappings(
        dsd, label_column, cfg["label_names"]
    )
    num_labels = len(label_names)
    print(f"num_labels={num_labels}  label2id={label_to_id}")

    config = AutoConfig.from_pretrained(
        args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task="text-classification",
        trust_remote_code=args.trust_remote_code,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        use_fast=True,
        trust_remote_code=args.trust_remote_code,
    )
    # AutoModelForSequenceClassification attaches a fresh classification head on
    # top of the encoder. ignore_mismatched_sizes lets it accept a checkpoint
    # that has no (or a differently sized) head.
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        config=config,
        trust_remote_code=args.trust_remote_code,
        ignore_mismatched_sizes=True,
    )

    # transformers >=5 validates label2id as dict[str, int] / id2label as dict[int, str].
    model.config.label2id = {str(n): i for i, n in enumerate(label_names)}
    model.config.id2label = {i: str(n) for i, n in enumerate(label_names)}

    # Some encoders (e.g. GPT-2 style) have no pad token; fall back to eos.
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id

    def preprocess(examples):
        tokenized = tokenizer(
            examples[text_column],
            truncation=True,
            max_length=args.max_seq_length,
        )
        if label_is_int:
            tokenized["labels"] = [int(l) for l in examples[label_column]]
        else:
            tokenized["labels"] = [label_to_id[str(l)] for l in examples[label_column]]
        return tokenized

    remove_cols = dsd["train"].column_names
    encoded = dsd.map(preprocess, batched=True, remove_columns=remove_cols,
                      desc="Tokenizing")

    train_dataset = encoded["train"]
    eval_dataset = encoded["validation"]
    test_dataset = encoded["test"]

    if args.max_train_samples is not None:
        train_dataset = train_dataset.select(
            range(min(args.max_train_samples, len(train_dataset))))
    if args.max_eval_samples is not None:
        eval_dataset = eval_dataset.select(
            range(min(args.max_eval_samples, len(eval_dataset))))
        test_dataset = test_dataset.select(
            range(min(args.max_eval_samples, len(test_dataset))))

    data_collator = DataCollatorWithPadding(tokenizer)

    def _macro_f1(preds, refs, n_labels):
        # Self-contained macro-F1 (no evaluate/sklearn dependency, so this runs
        # with just the project's base env: transformers + torch + datasets).
        f1s = []
        for c in range(n_labels):
            tp = int(np.sum((preds == c) & (refs == c)))
            fp = int(np.sum((preds == c) & (refs != c)))
            fn = int(np.sum((preds != c) & (refs == c)))
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = (2 * precision * recall / (precision + recall)
                  if (precision + recall) else 0.0)
            f1s.append(f1)
        return float(np.mean(f1s))

    def compute_metrics(p):
        preds = np.argmax(p.predictions, axis=1)
        refs = np.asarray(p.label_ids)
        accuracy = float(np.mean(preds == refs))
        return {"accuracy": accuracy, "macro_f1": _macro_f1(preds, refs, num_labels)}

    # NOTE: no `overwrite_output_dir` - removed/invalid in transformers 5.
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        do_train=True,
        do_eval=True,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        max_steps=args.max_steps,
        seed=args.seed,
        logging_strategy="epoch",
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        disable_tqdm=False,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,  # transformers 5: not `tokenizer=`
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    train_result = trainer.train()
    train_metrics = train_result.metrics
    train_metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", train_metrics)
    trainer.save_metrics("train", train_metrics)

    print("\nEvaluation,", args.model_name_or_path)
    val_metrics = trainer.evaluate(eval_dataset, metric_key_prefix="validation")
    test_metrics = trainer.evaluate(test_dataset, metric_key_prefix="test")
    print("validation:", val_metrics)
    print("test:", test_metrics)

    results = {
        "model_name_or_path": args.model_name_or_path,
        "dataset": args.dataset,
        "dataset_hub_id": cfg["hub_id"],
        "num_labels": num_labels,
        "label_names": label_names,
        "validation_accuracy": float(val_metrics.get("validation_accuracy", float("nan"))),
        "validation_macro_f1": float(val_metrics.get("validation_macro_f1", float("nan"))),
        "test_accuracy": float(test_metrics.get("test_accuracy", float("nan"))),
        "test_macro_f1": float(test_metrics.get("test_macro_f1", float("nan"))),
    }

    score = results["test_macro_f1"]
    print(f"\nModel: {args.model_name_or_path}, Dataset: {args.dataset}, "
          f"test_macro_f1: {score:.3f}, test_accuracy: {results['test_accuracy']:.3f}")

    save_path = Path(args.output_dir).resolve()
    save_path.mkdir(parents=True, exist_ok=True)
    (save_path / "textcls_results.json").write_text(json.dumps(results, indent=2))

    # Append a one-line summary, mirroring ner_bert.py's ner_results.tsv.
    with open("textcls_results.tsv", "a") as f:
        f.write(f"{args.model_name_or_path}\t{args.dataset}\t"
                f"{results['test_accuracy']:.3f}\t{score:.3f}\n")


if __name__ == "__main__":
    main()
