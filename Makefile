.DEFAULT_GOAL := help
# Silence MLflow's coding-agent hint printed on every import.
export MLFLOW_DISABLE_AGENT_HINT := 1
.PHONY: help setup lint format test test-fast loc ingest train models promote mlflow-ui serve monitor clean

help: ## List targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F ':.*## ' '{printf "  %-10s %s\n", $$1, $$2}'

setup: ## Install deps and git hooks
	uv sync
	uv run pre-commit install

lint: ## Check lint and formatting
	uv run ruff check .
	uv run ruff format --check .

format: ## Auto-fix lint and format code
	uv run ruff check --fix .
	uv run ruff format .

test: ## Run all tests (incl. integration: needs make ingest)
	uv run pytest -v

test-fast: ## Run tests except integration (no data needed)
	uv run pytest -v -m "not integration"

loc: ## Count code lines vs the ~800 budget
	uv run python scripts/loc.py

ingest: ## Download and split data
	uv run python -m credit_pd.data

train: ## Train and log a model
	uv run python -m credit_pd.train

models: ## List registered model versions
	uv run python -m credit_pd.registry list

promote: ## Set champion alias: make promote VERSION=n REASON="..."
	uv run python -m credit_pd.registry promote --version "$(VERSION)" --reason "$(REASON)"

# Backend URI must match mlflow.tracking_uri in config.yaml.
mlflow-ui: ## Open the MLflow UI on http://localhost:5000
	uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

serve: ## Run the scoring API
	@echo "not implemented (module 6)"

monitor: ## Check drift on future data
	@echo "not implemented (module 9)"

clean: ## Remove caches
	rm -rf .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -not -path "./.venv/*" -prune -exec rm -rf {} +
