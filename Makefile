.DEFAULT_GOAL := help
.PHONY: help setup lint format test loc ingest train serve monitor clean

help: ## List targets
	@grep -E '^[a-z]+:.*## ' $(MAKEFILE_LIST) | awk -F ':.*## ' '{printf "  %-10s %s\n", $$1, $$2}'

setup: ## Install deps and git hooks
	uv sync
	uv run pre-commit install

lint: ## Check lint and formatting
	uv run ruff check .
	uv run ruff format --check .

format: ## Auto-fix lint and format code
	uv run ruff check --fix .
	uv run ruff format .

test: ## Run tests
	uv run pytest

loc: ## Count code lines vs the ~800 budget
	uv run python scripts/loc.py

ingest: ## Download and split data
	uv run python -m credit_pd.data

train: ## Train and log a model
	uv run python -m credit_pd.train

serve: ## Run the scoring API
	@echo "not implemented (module 6)"

monitor: ## Check drift on future data
	@echo "not implemented (module 9)"

clean: ## Remove caches
	rm -rf .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -not -path "./.venv/*" -prune -exec rm -rf {} +
