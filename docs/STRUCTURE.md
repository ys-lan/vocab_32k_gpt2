# Repository Structure

## Goals

- Keep root directory focused on core training entrypoints and configs.
- Place executable scripts under `scripts/` by platform and function.
- Keep historical or experimental files under explicit archive paths.

## Layout Conventions

- `train.py`, `trainer.py`, `dpo.py`:
  Core runtime entrypoints used by training commands.
- `configs/`:
  All YAML/JSON configs for model, training, and accelerate/deepspeed.
- `dataset/`:
  Active data pipeline modules.
- `dataset/legacy/`:
  Archived data scripts not used by current training pipeline.
- `scripts/launch/`:
  Linux/WSL shell launch scripts.
- `scripts/eval/`:
  Inference/testing helper scripts.
- `logs/`:
  Local training logs (ignored by default for new `.out`/`.log` files).

## Operating Rules

- New launcher scripts should go into `scripts/launch/`, not root.
- New one-off experiments should be placed in `docs/` (notes) or `dataset/legacy/` (old data logic), not mixed into root.
- When moving files, update `README.md` paths in the same change.
