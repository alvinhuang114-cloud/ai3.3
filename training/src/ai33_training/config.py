"""Configuration loading and validation."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ai33_training.errors import PipelineError


@dataclass(frozen=True)
class PipelineConfig:
    model_id: str
    enable_thinking: bool
    train_file: Path
    eval_file: Path | None
    validation_split: float
    max_length: int
    lora_rank: int
    lora_alpha: int
    lora_dropout: float
    target_modules: tuple[str, ...]
    output_dir: Path
    epochs: float
    train_batch_size: int
    eval_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    weight_decay: float
    warmup_ratio: float
    logging_steps: int
    save_steps: int
    eval_steps: int
    save_total_limit: int
    seed: int
    gradient_checkpointing: bool
    max_retries: int


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PipelineError(f"Configuration section '{label}' must be an object.")
    return value


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise PipelineError(f"'{label}' must be a positive integer.")
    return value


def _number(value: Any, label: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PipelineError(f"'{label}' must be a number.")
    result = float(value)
    if not math.isfinite(result):
        raise PipelineError(f"'{label}' must be a finite number.")
    if result < minimum:
        raise PipelineError(f"'{label}' must be at least {minimum}.")
    return result


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PipelineError(f"'{label}' must be a non-negative integer.")
    return value


def _resolve_path(value: Any, label: str, base_dir: Path) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise PipelineError(f"'{label}' must be a non-empty path.")
    path = Path(value).expanduser()
    return (base_dir / path).resolve() if not path.is_absolute() else path.resolve()


def load_config(
    config_path: str | Path,
    *,
    train_file: str | Path | None = None,
    eval_file: str | Path | None = None,
) -> PipelineConfig:
    """Load a JSON config; relative data/output paths resolve from training/."""
    source = Path(config_path).expanduser().resolve()
    if not source.is_file():
        raise PipelineError(f"Configuration file does not exist: {source}")

    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PipelineError(
            f"Configuration JSON is invalid at line {exc.lineno}, column {exc.colno}."
        ) from None
    except OSError:
        raise PipelineError("Could not read the configuration file.") from None

    root = _mapping(raw, "root")
    model = _mapping(root.get("model"), "model")
    data = _mapping(root.get("data"), "data")
    lora = _mapping(root.get("lora"), "lora")
    training = _mapping(root.get("training"), "training")
    base_dir = source.parent.parent

    model_id = model.get("id")
    if not isinstance(model_id, str) or not model_id.strip():
        raise PipelineError("'model.id' must be a non-empty Hugging Face model ID.")
    thinking = model.get("enable_thinking", False)
    if not isinstance(thinking, bool):
        raise PipelineError("'model.enable_thinking' must be true or false.")

    actual_train_file = train_file if train_file is not None else data.get("train_file")
    actual_eval_file = eval_file if eval_file is not None else data.get("eval_file")
    resolved_train_file = _resolve_path(actual_train_file, "data.train_file", base_dir)
    resolved_eval_file = (
        _resolve_path(actual_eval_file, "data.eval_file", base_dir)
        if actual_eval_file
        else None
    )

    split = _number(data.get("validation_split", 0.1), "data.validation_split")
    if split >= 1:
        raise PipelineError("'data.validation_split' must be less than 1.")
    if resolved_eval_file is None and split <= 0:
        raise PipelineError(
            "Set a validation split greater than 0 or provide data.eval_file."
        )

    target_modules = lora.get("target_modules")
    if (
        not isinstance(target_modules, list)
        or not target_modules
        or any(not isinstance(item, str) or not item.strip() for item in target_modules)
    ):
        raise PipelineError("'lora.target_modules' must be a non-empty list of names.")

    gradient_checkpointing = training.get("gradient_checkpointing", True)
    if not isinstance(gradient_checkpointing, bool):
        raise PipelineError("'training.gradient_checkpointing' must be true or false.")

    dropout = _number(lora.get("dropout", 0.05), "lora.dropout")
    if dropout >= 1:
        raise PipelineError("'lora.dropout' must be less than 1.")
    warmup_ratio = _number(training.get("warmup_ratio", 0.03), "training.warmup_ratio")
    if warmup_ratio >= 1:
        raise PipelineError("'training.warmup_ratio' must be less than 1.")
    retries = training.get("max_retries", 2)
    if isinstance(retries, bool) or not isinstance(retries, int) or retries < 0 or retries > 5:
        raise PipelineError("'training.max_retries' must be an integer from 0 to 5.")

    return PipelineConfig(
        model_id=model_id.strip(),
        enable_thinking=thinking,
        train_file=resolved_train_file,
        eval_file=resolved_eval_file,
        validation_split=split,
        max_length=_positive_int(data.get("max_length", 512), "data.max_length"),
        lora_rank=_positive_int(lora.get("rank", 8), "lora.rank"),
        lora_alpha=_positive_int(lora.get("alpha", 16), "lora.alpha"),
        lora_dropout=dropout,
        target_modules=tuple(target_modules),
        output_dir=_resolve_path(
            training.get("output_dir", "outputs/qwen3-0.6b-lora"),
            "training.output_dir",
            base_dir,
        ),
        epochs=_number(training.get("epochs", 1), "training.epochs", minimum=0.01),
        train_batch_size=_positive_int(
            training.get("train_batch_size", 1), "training.train_batch_size"
        ),
        eval_batch_size=_positive_int(
            training.get("eval_batch_size", 1), "training.eval_batch_size"
        ),
        gradient_accumulation_steps=_positive_int(
            training.get("gradient_accumulation_steps", 8),
            "training.gradient_accumulation_steps",
        ),
        learning_rate=_number(
            training.get("learning_rate", 0.0002), "training.learning_rate", minimum=1e-12
        ),
        weight_decay=_number(
            training.get("weight_decay", 0.01), "training.weight_decay"
        ),
        warmup_ratio=warmup_ratio,
        logging_steps=_positive_int(
            training.get("logging_steps", 10), "training.logging_steps"
        ),
        save_steps=_positive_int(training.get("save_steps", 100), "training.save_steps"),
        eval_steps=_positive_int(training.get("eval_steps", 100), "training.eval_steps"),
        save_total_limit=_positive_int(
            training.get("save_total_limit", 2), "training.save_total_limit"
        ),
        seed=_nonnegative_int(training.get("seed", 42), "training.seed"),
        gradient_checkpointing=gradient_checkpointing,
        max_retries=retries,
    )
