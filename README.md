# AI3.3

AI3.3 is a model-development project. The training pipeline lives in `training/`
and is designed to run in a Kaggle notebook with a GPU; this Replit project does
not start training jobs.

## Quick start

1. Import
   [`training/notebooks/kaggle_qwen3_lora.ipynb`](https://raw.githubusercontent.com/alvinhuang114-cloud/ai3.3/main/training/notebooks/kaggle_qwen3_lora.ipynb)
   into Kaggle.
2. The setup cell reuses `/kaggle/working/ai3.3` if it already contains the
   project. It clones the public GitHub repository only when that folder is
   absent; it never clones over an existing folder or pulls updates by default.
3. Follow `training/README.md` to install dependencies and run the safe
   preflight. `START_TRAINING` defaults to `False`; training requires an
   intentional opt-in after checks pass.

The GitHub repository and Kaggle account must be accessible from the notebook.
Do not paste access tokens into notebook cells or commit them.

## Repository map

- `training/configs/` — model, data, LoRA, and training settings
- `training/src/ai33_training/` — validation, preflight, training, and evaluation
- `training/tests/` — lightweight local tests
- `training/notebooks/` — Kaggle entry point
