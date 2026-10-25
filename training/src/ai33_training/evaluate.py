"""Standalone evaluation of a saved LoRA adapter."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

from ai33_training.config import PipelineConfig, load_config
from ai33_training.data import CausalPaddingCollator, build_datasets
from ai33_training.errors import PipelineError
from ai33_training.runtime import (
    append_metrics,
    configure_logging,
    log_safe_failure,
    with_limited_retries,
)
from ai33_training.train import _check_requirements


def evaluate(
    config: PipelineConfig,
    checkpoint: Path,
    logger: Any,
) -> dict[str, Any]:
    torch, _, _ = _check_requirements()
    if not torch.cuda.is_available():
        raise PipelineError(
            "Evaluation requires a visible CUDA GPU. Select a Kaggle GPU accelerator."
        )
    if not checkpoint.is_dir():
        raise PipelineError(f"LoRA adapter directory does not exist: {checkpoint}")

    from peft import PeftModel
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    tokenizer = with_limited_retries(
        lambda: AutoTokenizer.from_pretrained(
            config.model_id, trust_remote_code=False, use_fast=True
        ),
        label="Tokenizer download",
        max_retries=config.max_retries,
        logger=logger,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    _, eval_data = build_datasets(config, tokenizer)
    dtype = (
        torch.bfloat16
        if torch.cuda.is_bf16_supported()
        else torch.float16
    )
    base_model = with_limited_retries(
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
    model = PeftModel.from_pretrained(base_model, str(checkpoint))
    args = TrainingArguments(
        output_dir=str(config.output_dir / "evaluation"),
        per_device_eval_batch_size=config.eval_batch_size,
        report_to=[],
        dataloader_num_workers=0,
    )
    trainer = Trainer(
        model=model,
        args=args,
        eval_dataset=eval_data,
        data_collator=CausalPaddingCollator(tokenizer),
        processing_class=tokenizer,
    )
    metrics = dict(trainer.evaluate())
    loss = metrics.get("eval_loss")
    if isinstance(loss, (int, float)):
        try:
            perplexity = float(torch.exp(torch.tensor(loss)).item())
            metrics["eval_perplexity"] = perplexity if math.isfinite(perplexity) else None
        except (OverflowError, RuntimeError):
            metrics["eval_perplexity"] = None
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)
    append_metrics(config.output_dir / "metrics.jsonl", {"stage": "eval", **metrics})
    logger.info("Evaluation completed. Metrics are under %s.", config.output_dir)
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a saved Qwen3 LoRA adapter.")
    parser.add_argument("--config", default="configs/qwen3_0_6b_lora.json")
    parser.add_argument(
        "--checkpoint",
        help="Adapter directory; defaults to <training.output_dir>/adapter.",
    )
    parser.add_argument("--eval-file", help="Override the configured evaluation JSONL file.")
    args = parser.parse_args(argv)
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path.cwd() / config_path

    logger = None
    try:
        config = load_config(config_path, eval_file=args.eval_file)
        checkpoint = (
            Path(args.checkpoint).expanduser().resolve()
            if args.checkpoint
            else config.output_dir / "adapter"
        )
        logger = configure_logging(config.output_dir / "logs", filename="evaluation.log")
        metrics = evaluate(config, checkpoint, logger)
        print(metrics)
        return 0
    except PipelineError as error:
        if logger:
            log_safe_failure(logger, error)
        else:
            print(f"Evaluation stopped: {error}", file=sys.stderr)
        return 2
    except Exception as error:
        if logger:
            log_safe_failure(logger, error)
        else:
            print(
                "Evaluation stopped: "
                + type(error).__name__
                + ". Check the adapter path and installed dependencies.",
                file=sys.stderr,
            )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
