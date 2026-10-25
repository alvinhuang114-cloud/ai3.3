# AI3.3

AI3.3 is a model-development project. The training pipeline lives in `training/`
and is designed to run in a Kaggle notebook with a GPU; this Replit project does
not start training jobs.

## Quick start

1. Open `training/notebooks/kaggle_qwen3_lora.ipynb` in Kaggle, or create a
   Kaggle notebook and clone this GitHub repository.
2. Attach a Kaggle GPU and add your JSONL dataset as a Kaggle input.
3. Follow `training/README.md`. The notebook defaults to a dry run. Training
   requires an explicit opt-in.

The GitHub repository and Kaggle account must be accessible from the notebook.
Do not paste access tokens into notebook cells or commit them.

## Repository map

- `training/configs/` — model, data, LoRA, and training settings
- `training/src/ai33_training/` — validation, preflight, training, and evaluation
- `training/tests/` — lightweight local tests
- `training/notebooks/` — Kaggle entry point
