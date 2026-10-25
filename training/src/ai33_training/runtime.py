"""Logging, limited retry handling, and safe error summaries."""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Any, Callable, TypeVar

from ai33_training.errors import PipelineError

T = TypeVar("T")
_SECRET_PATTERNS = (
    re.compile(r"\b(?:hf|gh[pousr])_[A-Za-z0-9_-]{8,}\b"),
    re.compile(r"(?i)(authorization\s*:\s*bearer\s+)[^\s,;]+"),
    re.compile(r"(?i)(access_token|token|password)=([^\s&]+)"),
)


def safe_error_message(error: BaseException, limit: int = 320) -> str:
    """Return a short error summary with common credential formats redacted."""
    message = f"{type(error).__name__}: {error}"
    message = re.sub(r"https?://[^/@\s]+:[^/@\s]+@", "https://[redacted]@", message)
    for pattern in _SECRET_PATTERNS:
        message = pattern.sub(lambda match: match.group(1) + "[redacted]" if match.lastindex else "[redacted]", message)
    message = " ".join(message.split())
    return message[:limit]


def configure_logging(output_dir: Path, *, filename: str = "training.log") -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("ai33_training")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%dT%H:%M:%S"
    )
    file_handler = logging.FileHandler(output_dir / filename, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def _is_recoverable_network_error(error: BaseException) -> bool:
    status = getattr(getattr(error, "response", None), "status_code", None)
    if status in {408, 425, 429, 500, 502, 503, 504}:
        return True
    name = type(error).__name__.lower()
    return any(part in name for part in ("timeout", "connection", "network", "temporar"))


def with_limited_retries(
    operation: Callable[[], T],
    *,
    label: str,
    max_retries: int,
    logger: logging.Logger,
) -> T:
    """Retry only transient-looking network failures; never retry training errors."""
    attempt = 0
    while True:
        try:
            return operation()
        except Exception as error:
            if not _is_recoverable_network_error(error) or attempt >= max_retries:
                logger.error("%s failed: %s", label, safe_error_message(error))
                raise PipelineError(
                    f"{label} failed: {safe_error_message(error)}"
                ) from None
            attempt += 1
            wait_seconds = min(2**attempt, 8)
            logger.warning(
                "%s had a temporary network error; retry %d/%d in %ds.",
                label,
                attempt,
                max_retries,
                wait_seconds,
            )
            time.sleep(wait_seconds)


def log_safe_failure(logger: logging.Logger, error: BaseException) -> None:
    logger.error("Pipeline stopped: %s", safe_error_message(error))


def append_metrics(path: Path, metrics: dict[str, Any]) -> None:
    """Append a compact JSON metrics record without including dataset contents."""
    import json
    from datetime import datetime, timezone

    safe_metrics = {
        str(key): value
        for key, value in metrics.items()
        if isinstance(value, (str, int, float, bool)) or value is None
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps(
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "metrics": safe_metrics,
                },
                sort_keys=True,
            )
            + "\n"
        )
