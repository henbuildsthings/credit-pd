"""Ops tests for registry.py against a throwaway MLflow (see `tmp_mlflow`)."""

import mlflow
import mlflow.sklearn
import pytest
from mlflow import MlflowClient
from sklearn.dummy import DummyClassifier

from credit_pd.registry import CHAMPION, list_versions, promote, setup_mlflow


def register_tiny_model(cfg: dict, git_dirty: str) -> int:
    """Register a DummyClassifier with the given git_dirty tag; return its version."""
    name = setup_mlflow(cfg)
    with mlflow.start_run():
        mlflow.set_tags({"git_dirty": git_dirty, "git_sha": "abc1234", "data_sha256": "f00"})
        model = DummyClassifier().fit([[0], [1]], [0, 1])
        info = mlflow.sklearn.log_model(model, name="model", registered_model_name=name)
    return info.registered_model_version


def test_list_shows_champion_alias(tmp_mlflow):
    """Regression test: `make models` once showed no alias while the champion was set."""
    version = register_tiny_model(tmp_mlflow, git_dirty="false")
    name = tmp_mlflow["mlflow"]["registered_model_name"]
    MlflowClient().set_registered_model_alias(name, CHAMPION, str(version))
    table = list_versions(tmp_mlflow)
    assert table.loc[version, "aliases"] == CHAMPION


def test_promote_refuses_dirty_version(tmp_mlflow):
    version = register_tiny_model(tmp_mlflow, git_dirty="true")
    with pytest.raises(SystemExit, match="uncommitted code"):
        promote(tmp_mlflow, version, reason="test")
    assert list_versions(tmp_mlflow).loc[version, "aliases"] == ""  # alias not moved
