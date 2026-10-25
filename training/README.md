# AI3.3 training pipeline

This pipeline fine-tunes the public `Qwen/Qwen3-0.6B` model with Hugging Face
Transformers and PEFT/LoRA. It is intended to run in Kaggle, where you select a
GPU accelerator yourself. Replit does not start, monitor, or control Kaggle
notebooks.

The pipeline is deliberately opt-in: running the training command without
`--run-training` only performs preflight checks. The included sample dataset is
for validating the workflow; it is not intended to train a useful model.

## Kaggle: phone-friendly flow

1. Review and push the project files to the connected GitHub repository using
   Replit's Git pane. This project does not commit or push changes automatically.
2. In Kaggle, create a Python notebook, enable a GPU in notebook settings, and
   attach your dataset using **Add Input**.
3. Open `notebooks/kaggle_qwen3_lora.ipynb` from the repository (or copy its
   cells into a new Kaggle notebook).
4. Set `REPO_URL` to your GitHub clone URL and run the repository setup cell.
   Public repositories need no credential. For a private repository, configure
   access through a secure GitHub/Kaggle credential flow; never put a token in a
   notebook cell, URL, output, or committed file.
5. Run the dependency cell, then run the dry-run cell. It checks the dataset,
   Python packages, CUDA visibility, and model configuration/tokenizer. It does
   not download model weights or start training.
6. To start training, set `TRAIN_FILE` to the attached JSONL file and
   `START_TRAINING = True`, then run the training cell. Keep the GPU accelerator
   enabled. Save a Kaggle notebook version after the run to preserve outputs.

Kaggle GPU access and session quotas are managed by Kaggle and can vary. Model
weights are downloaded from Hugging Face at run time. The public default model
does not require an API token.

## Dataset format

Use UTF-8 JSONL: one JSON object per line. Chat examples are recommended:

```json
{"messages":[{"role":"user","content":"What is a checkpoint?"},{"role":"assistant","content":"A checkpoint is a saved training state."}]}
```

The validator also accepts plain text records:

```json
{"text":"A checkpoint is a saved training state."}
```

For chat data, each row needs at least one non-empty `user` and `assistant`
message. Optional `system` messages are supported. The training loss is computed
over the rendered conversation tokens. Keep training and evaluation data
separate when possible; otherwise a deterministic held-out split is made.

## Commands

Run from the `training/` directory. Kaggle's notebook provides the GPU packages;
the local Replit project does not need a large PyTorch install.

```bash
# Install dependencies in Kaggle; this file intentionally does not install torch.
python -m pip install -r requirements-kaggle.txt

# Validate a dataset without downloading a model.
PYTHONPATH=src python -m ai33_training.validate_data /path/to/train.jsonl

# Preflight only (the default): no model weights, no training.
PYTHONPATH=src python -m ai33_training.train --dry-run

# Explicitly start training after attaching a GPU and providing your dataset.
PYTHONPATH=src python -m ai33_training.train \
  --run-training \
  --train-file /kaggle/input/your-dataset/train.jsonl \
  --eval-file /kaggle/input/your-dataset/validation.jsonl

# Evaluate the saved adapter.
PYTHONPATH=src python -m ai33_training.evaluate \
  --checkpoint outputs/qwen3-0.6b-lora/adapter \
  --eval-file /kaggle/input/your-dataset/validation.jsonl
```

When `--eval-file` is omitted, the configured validation split is taken from the
training file. Use `configs/qwen3_0_6b_lora.json` to adjust sequence length,
LoRA rank, batch size, learning rate, and checkpoint frequency. Reduce
`max_length` or batch size if the selected GPU runs out of memory.

## Outputs and recovery

- `outputs/qwen3-0.6b-lora/checkpoint-*` — automatic resumable checkpoints
- `outputs/qwen3-0.6b-lora/adapter/` — final LoRA adapter and tokenizer
- `outputs/qwen3-0.6b-lora/metrics.jsonl` — training/evaluation metrics
- `outputs/qwen3-0.6b-lora/logs/` — timestamped run log

If a prior checkpoint exists in the output directory, the next explicit training
run resumes from the latest checkpoint. Only temporary network failures during
model/tokenizer downloads are retried, with a bounded retry count and backoff.
Invalid data/configuration and GPU memory errors are reported without a retry
loop. Error logs omit dataset contents and redact common token formats.

## Local tests

The unit tests use Python's standard library and do not need a GPU or model
download:

```bash
cd training
PYTHONPATH=src python -m unittest discover -s tests -v
```
