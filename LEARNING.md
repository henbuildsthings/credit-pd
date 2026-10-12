# LEARNING.md

Running log of what was built, why, and what I should be able to explain.

## Decisions

- **Deployable unit = full sklearn Pipeline.** Preprocessing lives inside the pipeline; the API does no preprocessing of its own.
- **Split once, seeded, stratified: 55/15/15/15 train/val/test/future.** The "future" slice is never used in training.
- **Decision bands are policy.** They're kept in `config.yaml` and logged with each model.
- **Local MLflow is the registry of record.** The promoted model is exported as a GitHub Release asset for CI and Docker.
- **Every run logs** the git SHA, `uv.lock` hash, data SHA-256 and seed.
- **Dependency split: runtime = imported by `src/`, dev = only tools/tests.** Runtime: pandas, pyarrow, scikit-learn, pandera, mlflow, fastapi, uvicorn, pyyaml. Dev group (`uv add --dev`): pytest, httpx, ruff, pre-commit.
- **Large-file guard (500 KB) blocks data/models from git; `uv.lock` is excluded because it must be committed.**
- **Dev environment:** WSL2 Ubuntu, with the project in the Linux filesystem (not OneDrive or `/mnt/c`).

## Commands

<!-- Add commands per module as they're introduced. -->

## Interview questions

<!-- One subsection per module, e.g. "### Module 1 — Data". -->
