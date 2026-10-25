"""JSONL validation, chat rendering, and tokenization."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from ai33_training.config import PipelineConfig
from ai33_training.errors import PipelineError


class TokenizedDataset:
    """Small map-style dataset compatible with Transformers Trainer."""

    def __init__(self, rows: list[dict[str, list[int]]]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.rows[index]


def _normalize_record(record: Any, path: Path, line_number: int) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise PipelineError(f"{path.name}:{line_number}: each JSONL row must be an object.")

    has_text = "text" in record
    has_messages = "messages" in record
    if has_text == has_messages:
        raise PipelineError(
            f"{path.name}:{line_number}: provide exactly one of 'text' or 'messages'."
        )

    if has_text:
        text = record["text"]
        if not isinstance(text, str) or not text.strip():
            raise PipelineError(f"{path.name}:{line_number}: 'text' must not be empty.")
        return {"text": text.strip()}

    messages = record["messages"]
    if not isinstance(messages, list) or len(messages) < 2:
        raise PipelineError(
            f"{path.name}:{line_number}: 'messages' must contain at least two messages."
        )
    normalized: list[dict[str, str]] = []
    seen_roles: set[str] = set()
    for message in messages:
        if not isinstance(message, dict):
            raise PipelineError(
                f"{path.name}:{line_number}: every message must be an object."
            )
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str) or role not in {"system", "user", "assistant"}:
            raise PipelineError(
                f"{path.name}:{line_number}: message role must be system, user, or assistant."
            )
        if not isinstance(content, str) or not content.strip():
            raise PipelineError(
                f"{path.name}:{line_number}: message content must not be empty."
            )
        normalized.append({"role": role, "content": content.strip()})
        seen_roles.add(role)
    if "user" not in seen_roles or "assistant" not in seen_roles:
        raise PipelineError(
            f"{path.name}:{line_number}: chat rows need at least one user and assistant message."
        )
    return {"messages": normalized}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read and validate a JSONL file without including sample content in errors."""
    if not path.is_file():
        raise PipelineError(f"Dataset file does not exist: {path}")

    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise PipelineError(
                        f"{path.name}:{line_number}: invalid JSON "
                        f"(column {exc.colno})."
                    ) from None
                rows.append(_normalize_record(raw, path, line_number))
    except OSError:
        raise PipelineError(f"Could not read dataset file: {path}") from None

    if not rows:
        raise PipelineError(f"Dataset file contains no records: {path}")
    return rows


def validate_dataset(path: Path) -> int:
    """Validate a JSONL dataset and return its number of non-empty rows."""
    return len(read_jsonl(path))


def _render_record(
    record: dict[str, Any],
    tokenizer: Any,
    *,
    enable_thinking: bool,
) -> tuple[str, bool]:
    if "text" in record:
        return record["text"], False
    rendered = tokenizer.apply_chat_template(
        record["messages"],
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=enable_thinking,
    )
    return rendered, True


def _tokenize_records(
    records: list[dict[str, Any]],
    tokenizer: Any,
    *,
    max_length: int,
    enable_thinking: bool,
    source: Path,
) -> TokenizedDataset:
    rows: list[dict[str, list[int]]] = []
    for row_number, record in enumerate(records, start=1):
        text, is_chat = _render_record(
            record, tokenizer, enable_thinking=enable_thinking
        )
        encoded = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            add_special_tokens=not is_chat,
        )
        input_ids = encoded.get("input_ids")
        if not isinstance(input_ids, list) or len(input_ids) < 2:
            raise PipelineError(
                f"{source.name}: record {row_number} produces fewer than two tokens."
            )
        attention_mask = encoded.get("attention_mask", [1] * len(input_ids))
        rows.append(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }
        )
    return TokenizedDataset(rows)


def build_datasets(
    config: PipelineConfig,
    tokenizer: Any,
) -> tuple[TokenizedDataset, TokenizedDataset]:
    """Read train/eval JSONL, or make a deterministic held-out split."""
    train_rows = read_jsonl(config.train_file)
    if config.eval_file is not None:
        eval_rows = read_jsonl(config.eval_file)
    else:
        if len(train_rows) < 2:
            raise PipelineError(
                "At least two training rows are required to create an evaluation split."
            )
        indexes = list(range(len(train_rows)))
        random.Random(config.seed).shuffle(indexes)
        eval_count = max(1, int(len(indexes) * config.validation_split))
        eval_count = min(eval_count, len(indexes) - 1)
        eval_indexes = set(indexes[:eval_count])
        eval_rows = [row for index, row in enumerate(train_rows) if index in eval_indexes]
        train_rows = [
            row for index, row in enumerate(train_rows) if index not in eval_indexes
        ]

    train_data = _tokenize_records(
        train_rows,
        tokenizer,
        max_length=config.max_length,
        enable_thinking=config.enable_thinking,
        source=config.train_file,
    )
    eval_source = config.eval_file or config.train_file
    eval_data = _tokenize_records(
        eval_rows,
        tokenizer,
        max_length=config.max_length,
        enable_thinking=config.enable_thinking,
        source=eval_source,
    )
    return train_data, eval_data


class CausalPaddingCollator:
    """Pad examples and mask only padding positions (not EOS tokens)."""

    def __init__(self, tokenizer: Any):
        self.tokenizer = tokenizer

    def __call__(self, examples: list[dict[str, list[int]]]) -> dict[str, Any]:
        import torch

        batch = self.tokenizer.pad(examples, padding=True, return_tensors="pt")
        labels = batch["input_ids"].clone()
        labels[batch["attention_mask"] == 0] = -100
        batch["labels"] = labels
        return batch
