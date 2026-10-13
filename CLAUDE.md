# CLAUDE.md

Guardrails for Claude Code in this repo. This is a small learning project: a credit default (PD) scoring service on OpenML credit-g, built module by module to learn MLOps.

## Rules

- **Size budget:** total Python code (`src/` + `tests/`) must stay under ~800 lines. Report the line count after every change (e.g. `find src tests -name '*.py' | xargs wc -l`).
- **Dependencies:** never add a dependency without asking first.
- **Simplicity:** the simplest option wins unless told otherwise. No cloud services, Kubernetes, feature stores or orchestration frameworks.
- **Layout & tooling:** package is `src/credit_pd/`, Python 3.12, managed with `uv`. Run everything via `uv run` or `make` targets.
- **Config:** config values (seed, splits, decision bands) live in `config.yaml`, never hardcoded.
- **Learning log:** update `LEARNING.md` when a module finishes (decisions, commands, interview questions, lessons).
- **Scope:** only do the current module. Don't scaffold future modules.

## Environment

- WSL2 Ubuntu; project lives in the Linux filesystem (not OneDrive or `/mnt/c`).
- GitHub user: `henbuildsthings`. The repo will be public — never commit secrets, `.env` files or raw data.
