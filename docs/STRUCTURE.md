# Repository Structure

## Goals

- Keep the root directory focused on core training entrypoints and configs.
- Place executable scripts under `scripts/`, grouped by function.
- Keep historical or experimental files under explicit archive paths.

## Layout Conventions

- `train.py`, `trainer.py`, `dpo.py`:
  Core runtime entrypoints used by the training commands. `train.py` covers both
  pretraining and SFT, selected by `data.mode`; `dpo.py` covers preference alignment.
- `configs/`:
  All YAML/JSON configuration. Training configs live at the top level; architecture
  overrides in `model_configs/`; tokenizer artifacts in `tokenizer_models/`;
  accelerate/DeepSpeed presets in `accelerate_configs/`.
- `dataset/`:
  Active data pipeline modules.
- `dataset/legacy/`:
  Archived data scripts that are not imported by the current training pipeline.
- `models/`:
  Model, configuration and tokenizer classes. These files mirror upstream Hugging Face
  implementations closely and are excluded from the formatter so that diffs against
  upstream stay readable.
- `scripts/launch/`:
  Bash launch scripts, one per pipeline stage. Settings are read from the environment
  (`CUDA_VISIBLE_DEVICES`, `ACCELERATE_CONFIG`, `TRAIN_CONFIG`, `MODEL_CONFIG`) with
  defaults baked in, and trailing arguments are forwarded to the Python entrypoint.
- `scripts/eval/`:
  Inference and smoke-test helpers.
- `utils/`:
  Standalone maintenance tools that are not part of a training run.
- `logs/`:
  Local training logs. New `.out`/`.log` files are ignored by git.
- `docs/`:
  Design notes and conventions.

## Operating Rules

- New launcher scripts go into `scripts/launch/`, not the root.
- New one-off experiments go into `docs/` (notes) or `dataset/legacy/` (superseded data
  logic), never mixed into the root.
- Every user-facing script exposes a `--help` surface (`argparse`, `absl.flags` or
  `HfArgumentParser`) rather than requiring source edits to change paths.
- Prose, comments, docstrings and log messages are written in English. Non-English text is
  permitted only where it is training or evaluation *data*, such as the probe prompts in
  `dataset/validation.py` and the tokenizer vocabulary.
- Secrets are never committed. Credentials such as `WANDB_API_KEY` are read from the
  environment.
- When moving files, update `README.md` and this document in the same change.
