# CLAUDE.md

Guardrails for Claude Code in this repo. This is a small learning project: a credit default (PD) scoring service on OpenML credit-g, built module by module to learn MLOps.

## Rules

- **Size budget:** Python code in `src/` + `tests/` must stay under ~800 **code lines** (docstrings, comments and blank lines don't count). Report both numbers (code / all lines) after every change with `make loc`.
- **Dependencies:** never add a dependency without asking first.
- **Simplicity:** the simplest option wins unless told otherwise. No cloud services, Kubernetes, feature stores or orchestration frameworks.
- **Layout & tooling:** package is `src/credit_pd/`, Python 3.12, managed with `uv`. Run everything via `uv run` or `make` targets.
- **Config:** config values (seed, splits, decision bands) live in `config.yaml`, never hardcoded.
- **Learning log:** update `LEARNING.md` when a module finishes (decisions, commands, interview questions, lessons).
- **Documentation:** every module, class and function gets a docstring explaining what it does and why (design choices, edge cases, how it fits the pipeline). Add short comments for non-obvious lines only, not line by line. Applies to all new and changed code.
- **Scope:** only do the current module. Don't scaffold future modules.

## Environment

- WSL2 Ubuntu; project lives in the Linux filesystem (not OneDrive or `/mnt/c`).
- GitHub user: `henbuildsthings`. The repo will be public — never commit secrets, `.env` files or raw data.
