"""Model sanity on the real data: a performance floor, not a precise target.

Needs `make ingest` first; skipped otherwise. Reads data/processed/splits.parquet only;
no MLflow. Floors sit below the observed values by a noise margin (val AUC 0.743 with a
standard error of ~0.04 -> floor 0.70), so they catch a broken pipeline, not small changes.
"""

import pandas as pd
import pytest

from credit_pd.config import path
from credit_pd.metrics import evaluate
from credit_pd.train import build_pipelines, load_splits, xy

pytestmark = pytest.mark.integration

AUC_FLOOR = 0.70  # observed val AUC 0.743 minus a noise margin (SE ~0.04)
# Required lead over the prior-only baseline. Without a margin, a model trained on shuffled
# labels passed (0.503 > 0.500 is noise).
BASELINE_MARGIN = 0.10


@pytest.fixture
def val_metrics(cfg) -> dict[str, dict]:
    """Fit every configured model on train and evaluate it on val."""
    splits_path = path(cfg, "splits")
    if not splits_path.exists():
        pytest.skip(f"{splits_path.name} missing: run `make ingest` first")
    splits = load_splits(pd.read_parquet(splits_path))
    (X_train, y_train), (X_val, y_val) = xy(splits["train"]), xy(splits["val"])
    return {
        name: evaluate(y_val, pipe.fit(X_train, y_train).predict_proba(X_val)[:, 1])
        for name, pipe in build_pipelines(cfg, splits["train"]).items()
    }


def test_logreg_auc_above_floor(val_metrics):
    assert val_metrics["logreg"]["auc"] > AUC_FLOOR


def test_logreg_beats_baseline(val_metrics):
    assert val_metrics["logreg"]["auc"] > val_metrics["baseline"]["auc"] + BASELINE_MARGIN


def test_logreg_mean_pd_close_to_observed_rate(val_metrics):
    m = val_metrics["logreg"]
    assert abs(m["mean_pred"] - m["observed_rate"]) < 0.05
