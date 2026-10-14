"""Shared fixtures. Every test is offline, seeded, and isolated from the project's real
state: no network, no `data/`, and MLflow always points at a per-test temp folder."""

import mlflow
import numpy as np
import pandas as pd
import pytest

from credit_pd.config import load_config
from credit_pd.data import CreditSchema

# Range for integer columns that have no bounds in the schema (e.g. num_dependents).
DEFAULT_INT_RANGE = (1, 4)


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """60 schema-valid rows, generated from the schema itself, 18 defaults (30%).

    Categories are drawn from each column's `isin` values and integers from its bounds, so
    the fixture follows the schema automatically if the schema changes. Seeded, so every
    test sees the same rows.
    """
    rng = np.random.default_rng(0)
    n = 60
    data = {}
    for name, col in CreditSchema.to_schema().columns.items():
        stats = {k: v for check in col.checks for k, v in check.statistics.items()}
        if name == "default":
            data[name] = rng.permutation(np.r_[np.ones(18), np.zeros(n - 18)]).astype("int64")
        elif "allowed_values" in stats:
            data[name] = pd.array(rng.choice(stats["allowed_values"], n), dtype="str")
        else:
            lo = max(stats.get("min_value", DEFAULT_INT_RANGE[0]), 1)
            hi = min(stats.get("max_value", DEFAULT_INT_RANGE[1]), 10_000)
            data[name] = rng.integers(lo, hi + 1, n).astype("int64")
    return pd.DataFrame(data)


@pytest.fixture
def cfg() -> dict:
    """The real config.yaml, as a fresh dict per test (safe to modify)."""
    return load_config()


@pytest.fixture
def tmp_mlflow(cfg: dict, tmp_path) -> dict:
    """Config whose MLflow tracking DB and artifacts live in `tmp_path`.

    Pass the returned config to code under test. Each test gets an empty, throwaway
    registry, so tests can never register junk in, or move the champion of, the real
    `mlflow.db`. The global tracking URI is reset afterwards.
    """
    cfg["mlflow"]["tracking_uri"] = f"sqlite:///{tmp_path / 'mlflow.db'}"
    cfg["mlflow"]["artifact_dir"] = str(tmp_path / "mlruns")  # absolute: overrides ROOT
    yield cfg
    mlflow.set_tracking_uri(None)
