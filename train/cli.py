"""Command-line argument parsing for distillation.

Three parser sections are selected by name on the command line
(``data_params`` / ``model_params`` / ``training_params``); ``get_args`` splits
argv between them and returns the parsed (data_args, model_args, hf_training_args).
"""
import argparse
import ast
import logging
import os
import sys

import torch
from transformers import HfArgumentParser, TrainingArguments

logger = logging.getLogger(__name__)


def get_data_parser():
    parser = argparse.ArgumentParser(epilog="Data paths and configuration")
    parser.add_argument(
        "--dataset_name", type=str, default=None,
        help="Dataset to use (via the datasets library) if not loading local files.",
    )
    parser.add_argument(
        "--stream_local_files", action="store_true", default=True,
        help="If loading local Arrow files, try to stream them.",
    )
    parser.add_argument(
        "--pretokenized_dataset_path", type=str, default=None,
        help="Path to a pre-tokenized (pre-packed) dataset directory.",
    )
    parser.add_argument(
        "--dataset_config_name", type=str, default=None,
        help="Configuration name of the dataset to use.",
    )
    parser.add_argument(
        "--train_config", type=str, required=False,
        help="JSON config for the training dataset (local Arrow urls or Hub name/config).",
    )
    parser.add_argument(
        "--val_config", type=str, required=False,
        help="JSON config for the validation dataset.",
    )
    parser.add_argument(
        "--max_seq_len", type=int, default=512, help="Max sequence length for an input.",
    )
    parser.add_argument(
        "--pack", action="store_true",
        help="Pack documents into fixed-length rows with block-diagonal attention "
             "(near-zero padding, fixed shapes).",
    )
    return parser


def get_model_parser():
    parser = argparse.ArgumentParser(epilog="Model and distillation configuration")
    parser.add_argument(
        "--input_model_dir", type=str, required=True,
        help="Directory/id the pre-trained teacher is loaded from.",
    )
    parser.add_argument(
        "--tokenizer_dir", type=str, required=False,
        help="Tokenizer directory/id. Defaults to --input_model_dir.",
    )
    parser.add_argument(
        "--student_architecture", type=str, default=None,
        help="Config name/path for the student architecture (e.g. "
             "'answerdotai/ModernBERT-base'). Default: BERT (bert-base-uncased).",
    )
    parser.add_argument("--student_hidden_size", type=int, required=True, help="Student hidden size.")
    parser.add_argument("--student_num_layers", type=int, required=True, help="Student layer count.")
    parser.add_argument("--student_attention_heads", type=int, required=True, help="Student attention heads.")
    parser.add_argument(
        "--student_intermediate_size", type=int, default=None,
        help="Student FFN size. Default: scale the architecture's own "
             "intermediate/hidden ratio (4x for BERT, 1.5x GeGLU for ModernBERT).",
    )
    parser.add_argument("--L", type=int, required=True, help="Teacher's layer to distill from.")
    parser.add_argument("--num_relation_heads", type=int, required=True, help="Number of MiniLM relation heads.")
    parser.add_argument(
        "--head_chunk_size", type=int, default=8,
        help="Relation heads per loss chunk; bounds the peak memory of the "
             "(B, A_r, S, S) relation matrices. 0 disables chunking.",
    )
    parser.add_argument(
        "--teacher_dtype", type=str, choices=["auto", "bf16", "fp16", "fp32"], default="auto",
        help="Dtype for the frozen teacher. 'auto' = bf16 on CUDA, unchanged otherwise.",
    )
    parser.add_argument(
        "--prune_student_vocab", type=int, default=0,
        help="Prune the student's embedding table to the N most frequent token ids "
             "of the training corpus (0 = disabled). Teacher keeps the full vocab. "
             "For BPE tokenizers the set is then merge-closed so the exported "
             "tokenizer never emits an uncovered id; expect roughly +50-100%% rows.",
    )
    parser.add_argument(
        "--prune_vocab_coverage", type=float, default=0.0,
        help="Alternative to --prune_student_vocab: keep the smallest id set covering "
             "this fraction of corpus tokens (e.g. 0.995). Takes precedence if both set.",
    )
    parser.add_argument(
        "--vocab_sample_docs", type=int, default=50000,
        help="Documents sampled to estimate token frequencies for vocab pruning.",
    )
    parser.add_argument(
        "--compile_loss", action="store_true",
        help="torch.compile the relation loss (best with --pack: fixed shapes).",
    )
    parser.add_argument(
        "--compile_student", action="store_true",
        help="torch.compile the student forward/backward (best with --pack).",
    )
    parser.add_argument(
        "--compile_teacher", action="store_true",
        help="torch.compile the frozen teacher forward (best with --pack).",
    )
    parser.add_argument(
        "--no_truncate_teacher", action="store_true",
        help="Keep teacher layers above the distillation layer L (slower; debugging only).",
    )
    parser.add_argument(
        "--minilm_relations", type=str, required=False, default="{(1, 1): 1, (2, 2): 1, (3, 3): 1}",
        help="Relations and weights, as {(id1, id2): weight}. Relation ids: "
             "1=Query 2=Key 3=Value (content, pre-RoPE), 4=Query 5=Key (post-RoPE).",
    )
    parser.add_argument(
        "--rope_qk_weight", type=float, default=0.0,
        help="If >0, add a post-RoPE Q-K relation (4,5) with this weight (RoPE teacher+student).",
    )
    parser.add_argument(
        "--repr_weight", type=float, default=0.0,
        help="Weight of the hidden-state (TinyBERT-style) distillation term. 0 = off.",
    )
    parser.add_argument(
        "--logit_kd_weight", type=float, default=0.0,
        help="Weight of the output-logit KD term (KL over MLM logits). Requires teacher and "
             "student to share a vocabulary (no --prune_student_vocab) and runs the full teacher. 0 = off.",
    )
    parser.add_argument(
        "--logit_kd_temperature", type=float, default=2.0,
        help="Softmax temperature for --logit_kd_weight.",
    )
    parser.add_argument(
        "--logit_kd_seq_chunk", type=int, default=0,
        help="Process logit-KD in chunks of this many sequence positions to bound "
             "the fp32 softmax memory over a large vocab (0 = whole sequence).",
    )
    return parser


def split_args_by_parser(args, parsers):
    """Split a flat argv list between named sub-parsers, each introduced by its name."""
    parser_names = parsers.keys()
    parser_name_locs = sorted(
        ((name, args.index(name) if name in args else -1) for name in parser_names),
        key=lambda x: x[1],
    )
    assert all(loc != -1 for _, loc in parser_name_locs), f"Required keys {list(parser_names)} must be present"
    parsed = {}
    for idx, (name, loc) in enumerate(parser_name_locs):
        end = len(args) if idx == len(parser_name_locs) - 1 else parser_name_locs[idx + 1][1]
        parsed[name] = parsers[name].parse_args(args[loc + 1:end])
    return parsed


def is_distributed():
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def get_rank():
    local_rank_env = os.environ.get("LOCAL_RANK")
    if local_rank_env is not None:
        return int(local_rank_env)
    return int(os.environ.get("RANK", "0"))


def get_args():
    """Parse argv into (data_args, model_args, hf_training_args)."""
    parsers = {
        "data_params": get_data_parser(),
        "model_params": get_model_parser(),
        "training_params": HfArgumentParser((TrainingArguments)),
    }
    raw_cli_args = sys.argv[1:]
    params = split_args_by_parser(raw_cli_args, parsers)

    tp = params["training_params"]
    if getattr(tp, "label_names", None) == ["start_positions", "end_positions"]:
        logger.info("Removing default QA-style label_names for MiniLM distillation.")
        delattr(tp, "label_names")
    tp.dataloader_drop_last = True
    if "accelerator_config" in tp and not isinstance(tp.accelerator_config, dict):
        delattr(tp, "accelerator_config")

    hf_args = vars(tp)
    if "--optim" not in raw_cli_args and torch.cuda.is_available():
        hf_args["optim"] = "adamw_bnb_8bit"
        logger.info("Using 8-bit AdamW (adamw_bnb_8bit). Override with --optim.")
    if "--dataloader_persistent_workers" not in raw_cli_args and hf_args.get("dataloader_num_workers", 0) > 0:
        hf_args["dataloader_persistent_workers"] = True
    if hf_args.get("local_rank", -1) == -1:
        hf_args["local_rank"] = get_rank() if is_distributed() else -1

    hf_training_args = TrainingArguments(**hf_args)
    data_args = params["data_params"]
    model_args = params["model_params"]

    if getattr(model_args, "minilm_relations", None) and isinstance(model_args.minilm_relations, str):
        try:
            model_args.minilm_relations = ast.literal_eval(model_args.minilm_relations)
        except ValueError as e:
            raise ValueError(f"Could not parse minilm_relations '{model_args.minilm_relations}': {e}")
    return data_args, model_args, hf_training_args
