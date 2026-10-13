"""Model registry: MLflow setup, listing versions, promoting a champion, loading it.

Run with `make models`, `make promote VERSION=n REASON="..."`, or
`python -m credit_pd.registry {list,promote}`.

Local MLflow (SQLite backend, artifacts in `mlruns/`) is the registry of record. Every
`make train` registers a new version of the model but never deploys it: a version only
becomes the one in use when a human gives it the `champion` alias with `promote`. The
alias is a movable pointer, so rolling back is just promoting an older version again.
Each promotion is tagged with who, when, why and what it replaced, as an audit trail.
"""

import argparse
import subprocess
from datetime import UTC, datetime

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from sklearn.pipeline import Pipeline

from credit_pd.config import ROOT, load_config, path
from credit_pd.features import NON_FEATURES

CHAMPION = "champion"  # the alias the API loads (Module 6)


def setup_mlflow(cfg: dict) -> str:
    """Point MLflow at the project's tracking DB and experiment; return the model name.

    A relative SQLite path in the config is resolved from the repo root, so runs from any
    directory land in the same `mlflow.db`. The experiment is created on first use with an
    explicit artifact folder, so artifacts never end up wherever the command happened to
    be run from.
    """
    m = cfg["mlflow"]
    uri = m["tracking_uri"]
    if uri.startswith("sqlite:///") and not uri.startswith("sqlite:////"):
        uri = f"sqlite:///{ROOT / uri.removeprefix('sqlite:///')}"
    mlflow.set_tracking_uri(uri)
    if mlflow.get_experiment_by_name(m["experiment_name"]) is None:
        artifacts = (ROOT / m["artifact_dir"]).as_uri()
        mlflow.create_experiment(m["experiment_name"], artifact_location=artifacts)
    mlflow.set_experiment(m["experiment_name"])
    return m["registered_model_name"]


def git(*args: str) -> str:
    """Run a git command in the repo root and return its trimmed output."""
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def list_versions(cfg: dict) -> pd.DataFrame:
    """One row per registered version, newest first, with the facts needed to compare them.

    Metrics and reproducibility tags are read from each version's training run (the
    parent run that logged the model), so they are stored once, not copied onto versions.
    """
    name = setup_mlflow(cfg)
    client = MlflowClient()
    rows = []
    for v in client.search_model_versions(f"name = '{name}'"):
        run = client.get_run(v.run_id).data
        rows.append(
            {
                "version": int(v.version),
                "aliases": ",".join(v.aliases),
                "val_auc": run.metrics.get("val_auc"),
                "test_auc": run.metrics.get("test_auc"),
                "data_sha256": run.tags.get("data_sha256", "")[:12],
                "git_sha": run.tags.get("git_sha", "")[:7],
                "git_dirty": run.tags.get("git_dirty", "?"),
                "created": datetime.fromtimestamp(v.creation_timestamp / 1000).strftime(
                    "%Y-%m-%d %H:%M"
                ),
            }
        )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).sort_values("version", ascending=False).set_index("version")
    return df.round(4)


def promote(cfg: dict, version: int, reason: str) -> None:
    """Make `version` the champion, record why, then smoke-test the loaded champion.

    Refuses (exits non-zero, changes nothing) unless the version's training run has
    git_dirty=false: a model trained on uncommitted code can't be traced back to a commit,
    so it can't be reproduced. There is deliberately no override: commit and retrain.

    The version is tagged with promoted_by (git user.name), promoted_at (UTC), reason and
    previous_champion ("none" on the first promotion). The smoke check loads the model
    through the alias, exactly as the API will, and scores 3 test rows.
    """
    name = setup_mlflow(cfg)
    client = MlflowClient()
    run = client.get_run(client.get_model_version(name, str(version)).run_id)
    if run.data.tags.get("git_dirty") != "false":  # missing tag counts as dirty too
        raise SystemExit(
            f"Refusing to promote {name} v{version}: trained on uncommitted code "
            "(git_dirty is not false) — commit and retrain."
        )
    try:
        previous = client.get_model_version_by_alias(name, CHAMPION).version
    except MlflowException:
        previous = "none"
    client.set_registered_model_alias(name, CHAMPION, str(version))
    tags = {
        "promoted_by": git("config", "user.name"),
        "promoted_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "reason": reason,
        "previous_champion": previous,
    }
    for key, value in tags.items():
        client.set_model_version_tag(name, str(version), key, value)
    print(f"{name} v{version} is now {CHAMPION} (previous: {previous})")

    splits = pd.read_parquet(path(cfg, "splits"))
    sample = splits[splits["split"] == "test"].head(3)
    pd_scores = load_champion(cfg).predict_proba(sample.drop(columns=NON_FEATURES))[:, 1]
    print("Smoke check, 3 test rows:")
    print(pd.DataFrame({"pd": pd_scores.round(4), "actual_default": sample["default"].values}))


def load_champion(cfg: dict | None = None) -> Pipeline:
    """Load the current champion as the full sklearn Pipeline (raw rows in, PDs out)."""
    name = setup_mlflow(cfg or load_config())
    return mlflow.sklearn.load_model(f"models:/{name}@{CHAMPION}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="list all registered versions")
    p = sub.add_parser("promote", help="set the champion alias on a version")
    p.add_argument("--version", type=int, required=True)
    p.add_argument("--reason", required=True)
    args = parser.parse_args()
    cfg = load_config()
    if args.cmd == "list":
        versions = list_versions(cfg)
        print(versions.to_string() if len(versions) else "No registered versions yet.")
    else:
        promote(cfg, args.version, args.reason)
