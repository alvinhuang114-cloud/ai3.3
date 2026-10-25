"""Standalone JSONL dataset validation."""

from __future__ import annotations

import argparse
from pathlib import Path

from ai33_training.data import validate_dataset
from ai33_training.errors import PipelineError


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate an AI3.3 JSONL dataset.")
    parser.add_argument("path", type=Path, help="Path to a training or evaluation JSONL file.")
    args = parser.parse_args()
    try:
        count = validate_dataset(args.path.expanduser().resolve())
    except PipelineError as error:
        parser.exit(2, f"Dataset validation failed: {error}\n")
    print(f"Dataset valid: {count} records.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
