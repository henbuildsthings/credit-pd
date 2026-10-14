"""Training: fit candidate models, pick one on val, test it once, register it in MLflow.

Run with `make train` or `python -m credit_pd.train` (after `make ingest`).

Steps:
    1. Load the splits written by `make ingest`; drop the `future` rows straight away.
    2. Fit every model in `config.yaml` -> `models` on train, each as a full Pipeline
       (preprocessor + model), seeded from `seed`.
    3. Evaluate them all on val and pick the winner by val AUC.
    4. Evaluate only the winner on test, once. Test is the unbiased estimate of
       performance; looking at it for every candidate would turn it into a second val set.
    5. Log it all to MLflow and register the winner as a new model version.

MLflow layout: one parent run per `make train`, carrying the reproducibility tags (git
SHA, dirty flag, data hash, uv.lock hash, seed), the winner's metrics and the model. Each
candidate gets a nested child run with its params and val metrics. Registering does not
deploy: the new version has no alias until someone runs `make promote`.
"""

import hashlib
import sys

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from sklearn.base import ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from credit_pd.config import ROOT, load_config, path
from credit_pd.data import content_sha256
from credit_pd.features import NON_FEATURES, build_preprocessor
from credit_pd.metrics import evaluate
from credit_pd.registry import git, setup_mlflow

# Model names used in config.yaml -> sklearn classes. Config holds only the parameters.
MODEL_CLASSES: dict[str, type[ClassifierMixin]] = {
    "baseline": DummyClassifier,
    "logreg": LogisticRegression,
    "gbm": HistGradientBoostingClassifier,
}


def load_splits(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return {"train", "val", "test"} DataFrames. The `future` split is never returned."""
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


def run_tags(cfg: dict, data: pd.DataFrame) -> dict[str, str]:
    """The reproducibility contract: everything needed to recreate this exact model.

    Code (git SHA + whether there were uncommitted changes), data (content hash of the
    rows actually trained on), environment (hash of uv.lock) and randomness (seed). A dirty
    working tree means the SHA doesn't fully describe the code, so it prints a warning.

    `data` is the splits frame; without its `split` column it is exactly the ingested
    data, so its hash equals `content_sha256` in data/manifest.json for real runs.
    """
    dirty = bool(git("status", "--porcelain"))
    if dirty:
        warning = "WARNING: uncommitted changes; git_sha does not fully describe this code."
        print(warning, file=sys.stderr)
    return {
        "git_sha": git("rev-parse", "HEAD"),
        "git_dirty": str(dirty).lower(),
        "data_sha256": content_sha256(data.drop(columns="split")),
        "uv_lock_sha256": hashlib.sha256((ROOT / "uv.lock").read_bytes()).hexdigest(),
        "seed": str(cfg["seed"]),
    }


def scalars(metrics: dict, prefix: str) -> dict[str, float]:
    """Numeric entries of an `evaluate` result, prefixed for MLflow (e.g. `val_auc`)."""
    return {f"{prefix}_{k}": v for k, v in metrics.items() if isinstance(v, float)}


def calibration(metrics: dict) -> pd.DataFrame:
    """The calibration table of an `evaluate` result as a DataFrame indexed by decile."""
    return pd.DataFrame(metrics["calibration"]).set_index("bin")


def train(cfg: dict | None = None, data: pd.DataFrame | None = None) -> int:
    """Fit, compare on val, test the winner once, log + register in MLflow, print a report.

    Defaults to `config.yaml` and the splits file from `make ingest`. Tests pass their own
    config (pointing MLflow at a temporary folder) and a small splits frame instead.
    Returns the new registered version number.
    """
    cfg = cfg or load_config()
    data = pd.read_parquet(path(cfg, "splits")) if data is None else data
    model_name = setup_mlflow(cfg)
    splits = load_splits(data)
    X_train, y_train = xy(splits["train"])
    X_val, y_val = xy(splits["val"])

    with mlflow.start_run(run_name="train") as parent:
        mlflow.set_tags(run_tags(cfg, data))

        # Fit on train only; compare on val. predict_proba[:, 1] = P(default).
        val_metrics, fitted = {}, {}
        for name, pipe in build_pipelines(cfg, splits["train"]).items():
            with mlflow.start_run(run_name=name, nested=True):
                fitted[name] = pipe.fit(X_train, y_train)
                val_metrics[name] = evaluate(y_val, pipe.predict_proba(X_val)[:, 1])
                mlflow.log_params({"model": name, **cfg["models"][name]})
                mlflow.log_metrics(scalars(val_metrics[name], "val"))
        winner = max(val_metrics, key=lambda name: val_metrics[name]["auc"])
        model = fitted[winner]

        # The single look at test, for the winner only.
        X_test, y_test = xy(splits["test"])
        test_metrics = evaluate(y_test, model.predict_proba(X_test)[:, 1])

        mlflow.set_tag("winner", winner)
        mlflow.log_params({"model": winner, **cfg["models"][winner]})
        mlflow.log_metrics(scalars(val_metrics[winner], "val") | scalars(test_metrics, "test"))
        for split_name, m in (("val", val_metrics[winner]), ("test", test_metrics)):
            mlflow.log_text(calibration(m).to_csv(), f"calibration_{split_name}.csv")

        # Signature/example from raw features: the model's contract is raw rows in, and
        # (via pyfunc_predict_fn) class probabilities out, column 1 = P(default).
        info = mlflow.sklearn.log_model(
            model,
            name="model",
            signature=infer_signature(X_train, model.predict_proba(X_train)),
            input_example=X_train.head(3),
            pyfunc_predict_fn="predict_proba",
            registered_model_name=model_name,
        )

    cols = ["auc", "ks", "brier", "mean_pred"]
    comparison = pd.DataFrame(val_metrics).T[cols].astype(float)
    print("Validation (train-fitted models):")
    print(comparison.round(4).to_string(), end="\n\n")
    print(f"Winner by val AUC: {winner}\n")
    print("Test (winner only):")
    print(pd.Series({k: test_metrics[k] for k in cols}).round(4).to_string(), end="\n\n")
    print(f"Val calibration by predicted-PD decile ({winner}):")
    print(calibration(val_metrics[winner]).round(3).to_string())
    print(f"\nMLflow run: {parent.info.run_id}")
    print(f"Registered {model_name} version {info.registered_model_version} (no alias set)")
    return info.registered_model_version


if __name__ == "__main__":
    train()
