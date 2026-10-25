from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ai33_training.config import load_config
from ai33_training.data import build_datasets, read_jsonl, validate_dataset
from ai33_training.errors import PipelineError
from ai33_training.runtime import safe_error_message, with_limited_retries

TRAINING_DIR = Path(__file__).resolve().parents[1]


class DatasetTests(unittest.TestCase):
    def test_sample_dataset_is_valid(self) -> None:
        sample = TRAINING_DIR / "examples" / "sample_chat.jsonl"
        self.assertEqual(validate_dataset(sample), 4)

    def test_invalid_json_reports_line_without_record_content(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "broken.jsonl"
            path.write_text('{"text":"private sample"}\nnot-json\n', encoding="utf-8")
            with self.assertRaisesRegex(PipelineError, r"broken.jsonl:2: invalid JSON"):
                read_jsonl(path)

    def test_invalid_message_role_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad-role.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "messages": [
                            {"role": "user", "content": "question"},
                            {"role": "tool", "content": "not supported"},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(PipelineError, "role must be"):
                read_jsonl(path)

    def test_unhashable_message_role_is_reported_as_data_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad-role.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "messages": [
                            {"role": "user", "content": "question"},
                            {"role": ["assistant"], "content": "answer"},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(PipelineError, "role must be"):
                read_jsonl(path)

    def test_empty_dataset_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "empty.jsonl"
            path.write_text("\n  \n", encoding="utf-8")
            with self.assertRaisesRegex(PipelineError, "contains no records"):
                validate_dataset(path)

    def test_build_datasets_makes_train_and_eval_split(self) -> None:
        config = load_config(TRAINING_DIR / "configs" / "qwen3_0_6b_lora.json")

        class StubTokenizer:
            def apply_chat_template(self, messages, **kwargs):
                return "rendered conversation"

            def __call__(self, text, **kwargs):
                return {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}

        train_data, eval_data = build_datasets(config, StubTokenizer())
        self.assertEqual(len(train_data), 3)
        self.assertEqual(len(eval_data), 1)


class ConfigTests(unittest.TestCase):
    def test_config_resolves_sample_path_from_training_directory(self) -> None:
        config = load_config(TRAINING_DIR / "configs" / "qwen3_0_6b_lora.json")
        self.assertEqual(config.train_file, TRAINING_DIR / "examples" / "sample_chat.jsonl")
        self.assertEqual(config.model_id, "Qwen/Qwen3-0.6B")
        self.assertEqual(config.lora_rank, 8)


class RuntimeTests(unittest.TestCase):
    def test_common_hugging_face_tokens_are_redacted(self) -> None:
        message = safe_error_message(RuntimeError("failed with hf_abcdefgh12345678"))
        self.assertNotIn("hf_abcdefgh12345678", message)
        self.assertIn("[redacted]", message)

    def test_transient_network_failure_retries_then_succeeds(self) -> None:
        calls = 0

        def operation() -> str:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise ConnectionError("temporary")
            return "ok"

        import logging

        with patch("ai33_training.runtime.time.sleep"):
            result = with_limited_retries(
                operation,
                label="unit test",
                max_retries=1,
                logger=logging.getLogger("test"),
            )
        self.assertEqual(result, "ok")
        self.assertEqual(calls, 2)

    def test_non_recoverable_failure_does_not_retry(self) -> None:
        calls = 0

        def operation() -> None:
            nonlocal calls
            calls += 1
            raise ValueError("invalid input")

        import logging

        with self.assertRaises(PipelineError):
            with_limited_retries(
                operation,
                label="unit test",
                max_retries=3,
                logger=logging.getLogger("test"),
            )
        self.assertEqual(calls, 1)


if __name__ == "__main__":
    unittest.main()
