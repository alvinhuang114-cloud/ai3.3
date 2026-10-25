"""Preflight and explicit LoRA training entry point."""

from __future__ import annotations

import argparse
import importlib.util
import math
import sys
import tempfile
from pathlib import Path
from typing import Any

from ai33_training.config import PipelineConfig, load_config
from ai33_training.data import CausalPaddingCollator, build_datasets, validate_dataset
from ai33_training.errors import PipelineError
from ai33_training.runtime import (
    append_metrics,
    configure_logging,
    log_safe_failure,
    with_limited_retries,
)


def _check_requirements() -> tuple[Any, Any, Any]:
    missing = [
        package
        for module, package in (
            ("torch", "torch"),
            ("transformers", "transformers"),
            ("peft", "peft"),
            ("accelerate", "accelerate"),
        )
        if importlib.util.find_spec(module) is None
    ]
    if missing:
        raise PipelineError(
            "Missing training dependencies: "
            + ", ".join(missing)
            + ". Install training/requirements-kaggle.txt in Kaggle."
        )

    import torch
    import peft
    import transformers

    version = tuple(
        int(part) for part in transformers.__version__.split(".")[:2] if part.isdigit()
    )
    if version and version < (4, 51):
        raise PipelineError(
            "Transformers 4.51 or newer is required for Qwen3. "
            "Install training/requirements-kaggle.txt."
        )
    return torch, transformers, peft


def _check_writable_directory(path: Path) -> None:
    try:
        path.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path, prefix=".ai33-check-", delete=True):
            pass
    except OSError:
        raise PipelineError(f"Output directory is not writable: {path}") from None


def run_preflight(config: PipelineConfig, logger: Any) -> dict[str, Any]:
    """Check data, packages, model metadata, tokenizer, and runtime; load no weights."""
    torch, transformers, _ = _check_requirements()

    train_count = validate_dataset(config.train_file)
    if config.eval_file is not None:
        eval_count = validate_dataset(config.eval_file)
    else:
        if train_count < 2:
            raise PipelineError(
                "At least two training rows are required when validation_split is used."
            )
        eval_count = max(1, int(train_count * config.validation_split))
        eval_count = min(eval_count, train_count - 1)

    _check_writable_directory(config.output_dir)
    log_dir = config.output_dir / "logs"
    _check_writable_directory(log_dir)

    from transformers import AutoConfig, AutoTokenizer

    with_limited_retries(
        lambda: AutoConfig.from_pretrained(config.model_id, trust_remote_code=False),
        label="Hugging Face model configuration check",
        max_retries=config.max_retries,
        logger=logger,
    )
    tokenizer = with_limited_retries(
        lambda: AutoTokenizer.from_pretrained(
            config.model_id,
            trust_remote_code=False,
            use_fast=True,
        ),
        label="Hugging Face tokenizer check",
        max_retries=config.max_retries,
        logger=logger,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_data, eval_data = build_datasets(config, tokenizer)

    cuda_available = bool(torch.cuda.is_available())
    gpu_name = torch.cuda.get_device_name(0) if cuda_available else None
    report = {
        "model_id": config.model_id,
        "transformers_version": transformers.__version__,
        "train_records": len(train_data),
        "eval_records": len(eval_data),
        "cuda_available": cuda_available,
        "gpu_name": gpu_name,
        "weights_loaded": False,
        "training_started": False,
    }
    logger.info(
        "Preflight passed: model=%s, train_records=%d, eval_records=%d, CUDA=%s%s.",
        config.model_id,
        len(train_data),
        len(eval_data),
        cuda_available,
        f", GPU={gpu_name}" if gpu_name else "",
    )
    if not cuda_available:
        logger.warning(
            "No CUDA GPU is visible. Preflight can still pass, but training requires "
            "a Kaggle GPU accelerator."
        )
    logger.info("Model weights were not loaded and no training was started.")
    return report


def _select_precision(torch: Any) -> tuple[Any, bool, bool]:
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16, False, True
    return torch.float16, True, False


def _run_training(config: PipelineConfig, logger: Any) -> None:
    torch, _, _ = _check_requirements()
    if not torch.cuda.is_available():
        raise PipelineError(
            "Training requires a visible CUDA GPU. In Kaggle, select a GPU accelerator "
            "and rerun the notebook."
        )

    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
        get_last_checkpoint,
    )

    tokenizer = with_limited_retries(
        lambda: AutoTokenizer.from_pretrained(
            config.model_id,
            trust_remote_code=False,
            use_fast=True,
        ),
        label="Tokenizer download",
        max_retries=config.max_retries,
        logger=logger,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_data, eval_data = build_datasets(config, tokenizer)
    dtype, fp16, bf16 = _select_precision(torch)
    model = with_limited_retries(
        lambda: AutoModelForCausalLM.from_pretrained(
            config.model_id,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=False,
        ),
        label="Base model download",
        max_retries=config.max_retries,
        logger=logger,
    )
    model.config.use_cache = False
    model = get_peft_model(
        model,
        LoraConfig(
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=list(config.target_modules),
            task_type=TaskType.CAUSAL_LM,
            bias="none",
        ),
    )
    model.print_trainable_parameters()

    args = TrainingArguments(
        output_dir=str(config.output_dir),
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.train_batch_size,
        per_device_eval_batch_size=config.eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        logging_dir=str(config.output_dir / "logs"),
        logging_steps=config.logging_steps,
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        save_strategy="steps",
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        gradient_checkpointing=config.gradient_checkpointing,
        fp16=fp16,
        bf16=bf16,
        dataloader_num_workers=0,
        report_to=[],
        seed=config.seed,
        data_seed=config.seed,
        optim="adamw_torch",
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        data_collator=CausalPaddingCollator(tokenizer),
        processing_class=tokenizer,
    )

    checkpoint = get_last_checkpoint(str(config.output_dir))
    if checkpoint:
        logger.info("Resuming from the most recent checkpoint: %s", checkpoint)
    else:
        logger.info("Starting a new fine-tuning run.")

    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    adapter_dir = config.output_dir / "adapter"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    train_metrics = dict(train_result.metrics)
    trainer.log_metrics("train", train_metrics)
    trainer.save_metrics("train", train_metrics)
    append_metrics(config.output_dir / "metrics.jsonl", {"stage": "train", **train_metrics})

    eval_metrics = dict(trainer.evaluate())
    loss = eval_metrics.get("eval_loss")
    if isinstance(loss, (int, float)):
        try:
            perplexity = float(torch.exp(torch.tensor(loss)).item())
            eval_metrics["eval_perplexity"] = (
                perplexity if math.isfinite(perplexity) else None
            )
        except (OverflowError, RuntimeError):
            eval_metrics["eval_perplexity"] = None
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)
    append_metrics(config.output_dir / "metrics.jsonl", {"stage": "eval", **eval_metrics})
    logger.info(
        "Training finished. Adapter, checkpoints, logs, and metrics are under %s.",
        config.output_dir,
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the Qwen3 LoRA setup; training is opt-in."
    )
    parser.add_argument(
        "--config",
        default="configs/qwen3_0_6b_lora.json",
        help="Config path, relative to the training/ directory by default.",
    )
    parser.add_argument("--train-file", help="Override the configured JSONL training file.")
    parser.add_argument("--eval-file", help="Optional JSONL evaluation file.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--run-training",
        action="store_true",
        help="Start training. Requires a visible CUDA GPU.",
    )
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Run preflight checks only. This is the default.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path.cwd() / config_path

    logger = None
    try:
        config = load_config(
            config_path,
            train_file=args.train_file,
            eval_file=args.eval_file,
        )
        logger = configure_logging(config.output_dir / "logs")
        run_preflight(config, logger)
        if not args.run_training:
            print("Dry run complete. No model weights were loaded; training did not run.")
            return 0

        logger.info("Explicit training opt-in received.")
        _run_training(config, logger)
        return 0
    except PipelineError as error:
        message = str(error)
        if logger:
            log_safe_failure(logger, error)
        else:
            print(f"Pipeline stopped: {message}", file=sys.stderr)
        return 2
    except Exception as error:
        if logger:
            log_safe_failure(logger, error)
        else:
            print(
                "Pipeline stopped: "
                + type(error).__name__
                + ". Check the configuration and installed dependencies.",
                file=sys.stderr,
            )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
