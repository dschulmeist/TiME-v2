"""Entry point for MiniLMv2 distillation training.

Parses CLI args (see train/cli.py), builds the frozen teacher and the student,
wires the MiniLMDistiller into a Hugging Face Trainer, and runs. Trainer
callbacks live in train/callbacks.py. Run with:

    python -m train.distillation -- data_params ... model_params ... training_params ...
"""
import ast
import datetime
import json
import logging
import os
import sys
import traceback

import torch
import transformers
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForMaskedLM,
    AutoTokenizer,
    Trainer,
    set_seed,
)

from .callbacks import EfficiencyCallback, SaveStudentCallback
from .cli import get_args, is_distributed
from .data_pipeline import get_tokenized_datasets
from .distiller import MiniLMDistiller

_LOG_FORMAT = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"


def _configure_logging():
    """Attach a console handler to the 'train' package logger so all submodule
    logs (cli, callbacks, distillation) are emitted once."""
    pkg = logging.getLogger("train")
    pkg.setLevel(logging.INFO)
    if not any(isinstance(h, logging.StreamHandler) for h in pkg.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_LOG_FORMAT))
        pkg.addHandler(handler)


_configure_logging()
logger = logging.getLogger(__name__)


def _resolve_data_config(data_args):
    """Resolve the dataset source from --train_config / --val_config and CLI
    flags, setting the attributes the data pipeline reads (mutates data_args)."""
    train_config_path = getattr(data_args, "train_config", None)
    data_args.is_local_arrow_config = False
    data_args.local_arrow_files_config = None

    if train_config_path and os.path.exists(train_config_path):
        try:
            with open(train_config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            logger.info("Loaded train_config from: %s", train_config_path)
            data_args.text_column_name = cfg.get("text_column_name", getattr(data_args, "text_column_name", "text"))
            if cfg.get("format") == "arrow" and cfg.get("urls"):
                data_args.is_local_arrow_config = True
                data_args.local_arrow_files_config = cfg["urls"]
                data_args.dataset_name = None
                data_args.dataset_config_name = None
                logger.info("Configured for local Arrow files: %s", data_args.local_arrow_files_config)
            else:
                data_args.dataset_name = cfg.get("dataset_name", data_args.dataset_name)
                data_args.dataset_config_name = cfg.get("dataset_config_name", data_args.dataset_config_name)
                logger.info("Using Hub params: name='%s', config='%s'",
                            data_args.dataset_name, data_args.dataset_config_name)
        except Exception as e:
            logger.error("Failed to load train_config '%s': %s. Using CLI/defaults.",
                         train_config_path, e, exc_info=True)
    else:
        logger.info("train_config '%s' not found/provided. Using CLI/defaults.", train_config_path)

    data_args.text_column_name = getattr(data_args, "text_column_name", "text")
    data_args.shuffle_buffer_size = getattr(data_args, "shuffle_buffer_size", 10000)
    data_args.stream_take_size = getattr(data_args, "stream_take_size", 0)
    data_args.map_batch_size = getattr(data_args, "map_batch_size", 1000)

    if getattr(data_args, "streaming", False) and not data_args.is_local_arrow_config and \
            getattr(data_args, "dataset_name", None):
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "True"
        logger.info("HF_HUB_ENABLE_HF_TRANSFER=True for Hub streaming.")

    val_config_path = getattr(data_args, "val_config", None)
    data_args.val_config_content = None
    if val_config_path and os.path.exists(val_config_path):
        try:
            with open(val_config_path, "r", encoding="utf-8") as f:
                data_args.val_config_content = json.load(f)
        except Exception as e:
            logger.error("Could not load val_config '%s': %s", val_config_path, e)


def _load_teacher(input_model_dir, model_cls):
    """Load the frozen teacher. ModernBERT needs sdpa + no reference_compile so
    the (B, S, H) Q/K/V hook contract holds."""
    logger.info("Loading teacher from: %s", input_model_dir)
    teacher_config = AutoConfig.from_pretrained(input_model_dir, trust_remote_code=True)
    teacher_kwargs = {}
    if teacher_config.model_type == "modernbert":
        teacher_config.reference_compile = False
        teacher_kwargs = {"config": teacher_config, "attn_implementation": "sdpa"}
    teacher = model_cls.from_pretrained(input_model_dir, trust_remote_code=True, **teacher_kwargs)
    logger.info("Teacher loaded.")
    return teacher


def _build_student(model_args, teacher, model_cls, tokenizer_dir):
    """Build the student from its architecture template, sized per model_args and
    sharing the teacher's (pad-rounded) vocabulary. May resize the teacher's
    embeddings up to match. Returns the freshly initialized student."""
    if model_args.student_architecture:
        logger.info("Student architecture: %s", model_args.student_architecture)
        student_config = AutoConfig.from_pretrained(model_args.student_architecture, trust_remote_code=True)
    else:
        logger.info("No student architecture specified; using bert-base-uncased layout.")
        student_config = AutoConfig.from_pretrained("bert-base-uncased", trust_remote_code=True)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=True)
    # keep the architecture family's own FFN/hidden ratio unless overridden
    ffn_ratio = student_config.intermediate_size / student_config.hidden_size
    student_config.hidden_size = model_args.student_hidden_size
    student_config.num_hidden_layers = model_args.student_num_layers
    student_config.num_attention_heads = model_args.student_attention_heads
    student_config.intermediate_size = (
        model_args.student_intermediate_size or int(model_args.student_hidden_size * ffn_ratio)
    )
    # match the teacher's (often pad-rounded) embedding size, not just the
    # tokenizer length, so teacher/student vocabularies align row-for-row:
    # required for logit-KD and harmless otherwise (extra rows are unused).
    student_config.vocab_size = max(tokenizer.vocab_size, len(tokenizer), teacher.config.vocab_size)
    if student_config.vocab_size > teacher.config.vocab_size:
        logger.warning("Student vocab (%d) > teacher vocab (%d); resizing teacher embeddings.",
                       student_config.vocab_size, teacher.config.vocab_size)
        teacher.resize_token_embeddings(student_config.vocab_size)

    student_kwargs = {}
    if student_config.model_type == "modernbert":
        student_config.reference_compile = False
        student_kwargs = {"attn_implementation": "sdpa"}
        # layer_types must match the new depth; keep the family's global-every-n
        # pattern unless that would put the distillation layer M on a sliding layer
        n = student_config.global_attn_every_n_layers
        num_layers = model_args.student_num_layers
        layer_types = ["full_attention" if i % n == 0 else "sliding_attention" for i in range(num_layers)]
        if layer_types[num_layers - 1] != "full_attention":
            logger.info("Student distillation layer M=%d would be sliding-window; "
                        "using global attention in every student layer instead.", num_layers)
            student_config.global_attn_every_n_layers = 1
            layer_types = ["full_attention"] * num_layers
        student_config.layer_types = layer_types

    logger.info("Student configuration: %s", student_config)
    student = model_cls.from_config(student_config, trust_remote_code=True, **student_kwargs)
    logger.info("Student model initialized.")
    return student


def main():
    try:
        data_args, model_args, hf_training_args = get_args()

        logger.info("Python %s | torch %s | transformers %s (%s)",
                    sys.version.split()[0], torch.__version__,
                    transformers.__version__, transformers.__file__)
        logger.info("Run: rank=%s/%s device=%s n_gpu=%s output_dir=%s",
                    hf_training_args.process_index, hf_training_args.world_size,
                    hf_training_args.device, hf_training_args.n_gpu, hf_training_args.output_dir)

        _resolve_data_config(data_args)
        set_seed(hf_training_args.seed)

        if hf_training_args.process_index == 0:
            os.makedirs(hf_training_args.output_dir, exist_ok=True)
            file_handler = logging.FileHandler(os.path.join(hf_training_args.output_dir, "training.log"))
            file_handler.setFormatter(logging.Formatter(_LOG_FORMAT))
            pkg_logger = logging.getLogger("train")
            if not any(isinstance(h, logging.FileHandler) for h in pkg_logger.handlers):
                pkg_logger.addHandler(file_handler)

        input_model_dir = model_args.input_model_dir
        checkpoint_to_resume_from = hf_training_args.resume_from_checkpoint
        tokenizer_dir = model_args.tokenizer_dir or input_model_dir
        # logit-KD distils the teacher's MLM output distribution, so both models
        # need their LM head; otherwise the bare encoders suffice.
        model_cls = AutoModelForMaskedLM if model_args.logit_kd_weight > 0 else AutoModel

        teacher = _load_teacher(input_model_dir, model_cls)
        student = _build_student(model_args, teacher, model_cls, tokenizer_dir)

        logger.info("Loading and tokenizing datasets...")

        class TokenizerPathConfig:
            def __init__(self, path):
                self.tokenizer_name_or_path = path

        tokenizer_cfg_obj = TokenizerPathConfig(tokenizer_dir)
        prepacked_path = getattr(data_args, "pretokenized_dataset_path", None)
        prepacked = bool(prepacked_path)
        val_dataset = None
        if prepacked:
            from datasets import load_from_disk
            logger.info("Loading pre-packed dataset from %s (bypassing stream+tokenize+pack).", prepacked_path)
            train_dataset = load_from_disk(prepacked_path).with_format("torch")
            tokenizer_from_data_util = AutoTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=True)
        else:
            data_args.streaming = getattr(data_args, "streaming", True)
            data_args.do_eval = getattr(data_args, "do_eval", False)
            data_args.preprocessing_num_workers = getattr(data_args, "preprocessing_num_workers", None)
            data_args.overwrite_cache = getattr(data_args, "overwrite_cache", False)
            data_args.stream_take_size_eval = getattr(data_args, "stream_take_size_eval", 0)
            train_dataset, val_dataset, tokenizer_from_data_util = get_tokenized_datasets(
                data_args, tokenizer_cfg_obj, hf_training_args
            )

        student_input_remap = None
        if model_args.prune_student_vocab or model_args.prune_vocab_coverage:
            from .vocab_pruning import (
                apply_vocab_pruning, count_token_ids, merge_closure,
                select_vocab, select_vocab_by_coverage,
            )
            logger.info("Sampling %d documents for vocab pruning...", model_args.vocab_sample_docs)
            counts = count_token_ids(iter(train_dataset), model_args.vocab_sample_docs)
            unk_id = tokenizer_from_data_util.unk_token_id
            if unk_id is None:
                unk_id = tokenizer_from_data_util.pad_token_id
            must_keep = set(tokenizer_from_data_util.all_special_ids) | {unk_id}
            if model_args.prune_vocab_coverage:
                keep_ids = select_vocab_by_coverage(counts, model_args.prune_vocab_coverage, sorted(must_keep))
            else:
                keep_ids = select_vocab(counts, model_args.prune_student_vocab, sorted(must_keep))
            # BPE: cover merge intermediates so export can prune merges without
            # changing segmentation and the embedding has a row for every piece
            keep_ids = merge_closure(keep_ids, tokenizer_from_data_util)
            student_input_remap = apply_vocab_pruning(student, keep_ids, unk_id)
            if hf_training_args.process_index == 0:
                os.makedirs(hf_training_args.output_dir, exist_ok=True)
                with open(os.path.join(hf_training_args.output_dir, "vocab_map.json"), "w") as f:
                    json.dump({"keep_ids": keep_ids, "unk_id": unk_id}, f)

        if prepacked or getattr(data_args, "pack", False):
            # a pre-packed cache is already packed; only pack on the fly when live
            if not prepacked and getattr(data_args, "pack", False):
                from .packing import pack_dataset
                logger.info("Packing documents into fixed-length rows.")
                train_dataset = pack_dataset(
                    train_dataset, data_args.max_seq_len, tokenizer_from_data_util.pad_token_id
                )
                if val_dataset is not None:
                    val_dataset = pack_dataset(
                        val_dataset, data_args.max_seq_len, tokenizer_from_data_util.pad_token_id
                    )
            data_collator = transformers.default_data_collator
        else:
            data_collator = transformers.DataCollatorWithPadding(tokenizer_from_data_util, padding="longest")

        logger.info("Initializing MiniLMv2 distiller...")
        relations = model_args.minilm_relations
        if not isinstance(relations, dict):
            relations = ast.literal_eval(str(relations))
        if getattr(model_args, "rope_qk_weight", 0.0) > 0:
            relations = {**relations, (4, 5): model_args.rope_qk_weight}
        logger.info("Teacher model_type=%s, relations=%s", teacher.config.model_type, relations)

        teacher_dtype = {
            "auto": None, "bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32,
        }[model_args.teacher_dtype]
        distiller = MiniLMDistiller(
            teacher=teacher, student=student, L=model_args.L, M=model_args.student_num_layers,
            relations=relations, A_r=model_args.num_relation_heads,
            teacher_dtype=teacher_dtype,
            truncate_teacher=not model_args.no_truncate_teacher,
            head_chunk_size=model_args.head_chunk_size or None,
            compile_loss=model_args.compile_loss,
            compile_student=model_args.compile_student,
            compile_teacher=model_args.compile_teacher,
            student_input_remap=student_input_remap,
            repr_weight=model_args.repr_weight,
            logit_kd_weight=model_args.logit_kd_weight,
            logit_kd_temperature=model_args.logit_kd_temperature,
            logit_kd_seq_chunk=model_args.logit_kd_seq_chunk or None,
        )

        if is_distributed() and os.environ.get("TOKENIZERS_PARALLELISM", "true").lower() == "true":
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
        if hf_training_args.process_index == 0:
            transformers.utils.logging.set_verbosity_info()

        # for a non-PreTrainedModel (our distiller) transformers 5.11's Trainer._save
        # always uses safetensors regardless of this flag, so it is effectively a
        # no-op here. safetensors refuses shared tensors, and a tied LM head
        # (logit-KD) shares decoder/embedding storage: that is why
        # MiniLMDistiller.state_dict() clones tensors (breaking the tie) and drops
        # the teacher. The exported student is saved by SaveStudentCallback.
        hf_training_args.remove_unused_columns = False
        hf_training_args.save_safetensors = False
        efficiency_meta = {
            "teacher": input_model_dir,
            "student_params_m": round(sum(p.numel() for p in student.parameters()) / 1e6, 1),
            "relations": str(relations),
            "repr_weight": model_args.repr_weight,
            "logit_kd_weight": model_args.logit_kd_weight,
            "teacher_truncated": not (model_args.no_truncate_teacher or model_args.logit_kd_weight > 0),
            "batch": hf_training_args.per_device_train_batch_size,
            "grad_accum": hf_training_args.gradient_accumulation_steps,
            "seq_len": data_args.max_seq_len,
            "world_size": hf_training_args.world_size,
        }
        trainer = Trainer(
            model=distiller,
            args=hf_training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            processing_class=tokenizer_from_data_util,
            data_collator=data_collator,
            callbacks=[SaveStudentCallback(), EfficiencyCallback(efficiency_meta)],
        )

        logger.info("Starting training (resume_from_checkpoint=%s).", checkpoint_to_resume_from or "auto")
        trainer.train(resume_from_checkpoint=checkpoint_to_resume_from)
        logger.info("---- TRAINING COMPLETED ----")

        if hf_training_args.save_strategy != "no" and hf_training_args.process_index == 0:
            logger.info("Saving final model + state to %s", hf_training_args.output_dir)
            trainer.save_model()
            trainer.save_state()

    except Exception as e:
        errors_dir = "errors"
        os.makedirs(errors_dir, exist_ok=True)
        stamp = str(datetime.datetime.now()).replace(" ", "_").replace(":", "-")
        filename = os.path.join(errors_dir, f"error_{stamp}.txt")
        with open(filename, "w") as f:
            f.write(f"Error in {__file__} (PID: {os.getpid()}):\n{e}\n\n")
            f.write(traceback.format_exc())
        logger.error("An error occurred: %s. Full traceback saved to %s", e, filename, exc_info=True)
        raise


if __name__ == "__main__":
    main()
