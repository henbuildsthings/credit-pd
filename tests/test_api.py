"""API tests: FastAPI TestClient against a throwaway MLflow holding a real champion."""

import json

import mlflow
import mlflow.sklearn
import pytest
from fastapi.testclient import TestClient
from mlflow import MlflowClient
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from credit_pd.api import band, create_app
from credit_pd.features import build_preprocessor
from credit_pd.registry import CHAMPION, setup_mlflow


@pytest.fixture
def champion(tmp_mlflow, sample_df) -> Pipeline:
    """Fit a small Pipeline on sample_df, register it and make it champion."""
    X, y = sample_df.drop(columns="default"), sample_df["default"]
    model = Pipeline([("prep", build_preprocessor(sample_df)), ("model", LogisticRegression())])
    model.fit(X, y)
    name = setup_mlflow(tmp_mlflow)
    with mlflow.start_run():
        info = mlflow.sklearn.log_model(model, name="model", registered_model_name=name)
    MlflowClient().set_registered_model_alias(name, CHAMPION, str(info.registered_model_version))
    return model


@pytest.fixture
def client(tmp_mlflow, champion):
    with TestClient(create_app(tmp_mlflow)) as c:  # `with` runs the startup (lifespan)
        yield c


@pytest.fixture
def applicant(sample_df) -> dict:
    """First sample row as plain JSON types (no numpy ints), without the target."""
    return json.loads(sample_df.drop(columns="default").iloc[[0]].to_json(orient="records"))[0]


def test_health_reports_model_version(client):
    assert client.get("/health").json() == {"status": "ok", "model_version": 1}


def test_health_503_without_champion(tmp_mlflow):
    with TestClient(create_app(tmp_mlflow)) as c:
        assert c.get("/health").status_code == 503


def test_score_valid_applicant(client, applicant):
    body = client.post("/score", json=applicant).json()
    assert 0 <= body["pd"] <= 1
    assert body["band"] in {"approve", "review", "decline"}
    assert (body["model_name"], body["model_version"]) == ("credit-pd", 1)


@pytest.mark.parametrize(
    "change",
    [
        pytest.param(lambda a: a | {"age": 999}, id="age-out-of-range"),
        pytest.param(lambda a: a | {"housing": "boat"}, id="unknown-category"),
        pytest.param(lambda a: a | {"extra": 1}, id="extra-field"),
        pytest.param(lambda a: {k: v for k, v in a.items() if k != "age"}, id="missing-field"),
    ],
)
def test_invalid_input_is_422(client, applicant, change):
    assert client.post("/score", json=change(applicant)).status_code == 422


@pytest.mark.parametrize(
    "pd_value, expected", [(0.1499, "approve"), (0.15, "review"), (0.40, "decline")]
)
def test_band_boundaries(cfg, pd_value, expected):
    assert band(pd_value, cfg["bands"]) == expected


def test_api_matches_pipeline(client, champion, sample_df, applicant):
    """Training/serving parity: the API returns the Pipeline's own PD (to 4 decimals)."""
    expected = champion.predict_proba(sample_df.drop(columns="default").iloc[[0]])[0, 1]
    assert client.post("/score", json=applicant).json()["pd"] == pytest.approx(expected, abs=1e-4)
