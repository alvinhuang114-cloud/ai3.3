# AI3.3 training pipeline

This pipeline fine-tunes the public `Qwen/Qwen3-0.6B` model with Hugging Face
Transformers and PEFT/LoRA. It is intended to run in Kaggle, where you select a
GPU accelerator yourself. Replit does not start, monitor, or control Kaggle
notebooks.

The pipeline is deliberately opt-in: running the training command without
`--run-training` only performs preflight checks. The included sample dataset is
for validating the workflow; it is not intended to train a useful model.

## Kaggle: phone-friendly setup and dry run

1. In Kaggle, import the notebook file from GitHub or download it and use
   Kaggle's notebook import/upload flow:
   [`kaggle_qwen3_lora.ipynb`](https://raw.githubusercontent.com/alvinhuang114-cloud/ai3.3/main/training/notebooks/kaggle_qwen3_lora.ipynb).
2. The setup cell checks `/kaggle/working/ai3.3`. If the complete project is
   already there, it uses that copy without cloning or pulling. If the folder is
   absent, it clones
   `https://github.com/alvinhuang114-cloud/ai3.3.git`. If the folder exists but
   is incomplete, setup stops with a message instead of cloning over it.
3. To intentionally update an existing Git checkout, first make sure it has no
   local changes, then set `UPDATE_EXISTING_REPO = True` in the setup cell and
   run that cell. The update is fast-forward-only and only runs when the
   checkout's `origin` matches this AI3.3 repository. Leave it `False` for
   normal setup. If the folder is a downloaded copy without Git metadata, update
   it separately; the notebook will not replace it.
4. Run the dependency cell. Kaggle's existing CUDA-enabled PyTorch is left
   untouched; the remaining packages are installed from
   `requirements-kaggle.txt`.
5. Keep `START_TRAINING = False`. Leave `TRAIN_FILE` empty to validate the
   bundled sample, or set it to the path of an attached JSONL dataset to check
   that dataset. Run the preflight cell to check dependencies, data, GPU
   visibility, model configuration, and tokenizer. It does **not** load model
   weights or start training, but it needs internet access to fetch the model
   configuration and tokenizer files.
6. Check the preflight report for `cuda_available` and the GPU name. A CPU-only
   dry run can still pass with a warning; training requires a visible Kaggle
   GPU. Keep `START_TRAINING = False` until all checks pass.

To update the notebook itself after a project change, import the latest `.ipynb`
from the link above into Kaggle. The notebook and the code folder are separate:
refresh the code folder as described in step 3 when you need the latest training
code. Do not paste credentials into notebook cells, URLs, output, or committed
files.

Training remains an explicit, separate action: it requires setting `TRAIN_FILE`
and changing `START_TRAINING` to `True`. Save a Kaggle notebook version or
download outputs after a run; `/kaggle/working` is the configured working area.

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
