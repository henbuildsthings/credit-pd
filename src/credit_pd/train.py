"""Training: fit candidate models, pick one on val, test it once, save it.

Run with `make train` or `python -m credit_pd.train` (after `make ingest`).

Steps:
    1. Load the splits written by `make ingest`; drop the `future` rows straight away.
    2. Fit every model in `config.yaml` -> `models` on train, each as a full Pipeline
       (preprocessor + model), seeded from `seed`.
    3. Evaluate them all on val and pick the winner by val AUC.
    4. Evaluate only the winner on test, once. Test is the unbiased estimate of
       performance; looking at it for every candidate would turn it into a second val set.
    5. Save the winning Pipeline to `models/model.joblib` and all metrics to
       `reports/metrics.json` (both gitignored; MLflow takes over in Module 4).
"""

import json

import joblib
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from credit_pd.config import load_config, path
from credit_pd.features import NON_FEATURES, build_preprocessor
from credit_pd.metrics import evaluate

# Model names used in config.yaml -> sklearn classes. Config holds only the parameters.
MODEL_CLASSES: dict[str, type[ClassifierMixin]] = {
    "baseline": DummyClassifier,
    "logreg": LogisticRegression,
    "gbm": HistGradientBoostingClassifier,
}


def load_splits(cfg: dict) -> dict[str, pd.DataFrame]:
    """Return {"train", "val", "test"} DataFrames. The `future` split is never returned."""
    df = pd.read_parquet(path(cfg, "splits"))
    return {name: df[df["split"] == name] for name in ("train", "val", "test")}


def build_pipelines(cfg: dict, train: pd.DataFrame) -> dict[str, Pipeline]:
    """One unfitted Pipeline (preprocessor + model) per entry in `cfg["models"]`.

    Every model gets `random_state=seed`, so reruns give identical results. (It only
    matters for some models, e.g. gradient boosting, but setting it everywhere is simpler
    than tracking which ones use it.)
    """
    return {
        name: Pipeline(
            [
                ("prep", build_preprocessor(train)),
                ("model", MODEL_CLASSES[name](**params, random_state=cfg["seed"])),
            ]
        )
        for name, params in cfg["models"].items()
    }


def xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split a splits-frame into features X (raw columns) and target y (`default`)."""
    return df.drop(columns=NON_FEATURES), df["default"]


def train() -> None:
    """Fit, compare on val, test the winner once, save model + metrics, print a report."""
    cfg = load_config()
    splits = load_splits(cfg)
    X_train, y_train = xy(splits["train"])
    X_val, y_val = xy(splits["val"])

    # Fit on train only; compare on val. predict_proba[:, 1] = P(default).
    val_metrics = {}
    fitted = {}
    for name, pipe in build_pipelines(cfg, splits["train"]).items():
        fitted[name] = pipe.fit(X_train, y_train)
        val_metrics[name] = evaluate(y_val, pipe.predict_proba(X_val)[:, 1])
    winner = max(val_metrics, key=lambda name: val_metrics[name]["auc"])

    # The single look at test, for the winner only.
    X_test, y_test = xy(splits["test"])
    test_metrics = evaluate(y_test, fitted[winner].predict_proba(X_test)[:, 1])

    model_path, metrics_path = path(cfg, "model"), path(cfg, "metrics")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted[winner], model_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    report = {"winner": winner, "val": val_metrics, "test": test_metrics}
    metrics_path.write_text(json.dumps(report, indent=2) + "\n")

    cols = ["auc", "ks", "brier", "mean_pred"]
    comparison = pd.DataFrame(val_metrics).T[cols].astype(float)
    print("Validation (train-fitted models):")
    print(comparison.round(4).to_string(), end="\n\n")
    print(f"Winner by val AUC: {winner}\n")
    print("Test (winner only):")
    print(pd.Series({k: test_metrics[k] for k in cols}).round(4).to_string(), end="\n\n")
    print(f"Val calibration by predicted-PD decile ({winner}):")
    print(pd.DataFrame(val_metrics[winner]["calibration"]).set_index("bin").round(3).to_string())
    print(f"\nSaved {model_path.name} and {metrics_path.name}")


if __name__ == "__main__":
    train()
