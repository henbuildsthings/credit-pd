"""Scoring API: raw applicant in, PD + decision band out.

Run with `make serve` (uvicorn on 0.0.0.0:8000); interactive docs at /docs.

- The request model is generated from `data.FEATURES`, the same contract the pandera
  schema uses, so the API accepts exactly what training accepted: known categories only,
  integer bounds, no extra or missing fields. Anything else is a 422 before the model runs.
- The model is the champion's full sklearn Pipeline, so the API does no preprocessing of
  its own: it only builds a one-row DataFrame with the training column order and dtypes.
- The decision band is policy from `config.yaml` (`bands`), applied on top of the PD.
"""

import logging
from contextlib import asynccontextmanager
from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from mlflow.exceptions import MlflowException
from pydantic import ConfigDict, Field, create_model
from sklearn.pipeline import Pipeline

from credit_pd.config import load_config
from credit_pd.data import DTYPES, FEATURES
from credit_pd.registry import load_champion

log = logging.getLogger(__name__)

# One field per feature: Literal of the allowed values for categoricals, int with the same
# bounds for integers. extra="forbid" rejects unknown fields instead of silently dropping them.
Applicant = create_model(
    "Applicant",
    __config__=ConfigDict(extra="forbid"),
    **{
        name: (Literal[tuple(spec)], ...) if isinstance(spec, list) else (int, Field(**spec))
        for name, spec in FEATURES.items()
    },
)


def band(pd_value: float, bands: dict) -> str:
    """Map a PD to approve / review / decline using the thresholds in `config.yaml`."""
    if pd_value < bands["approve_below"]:
        return "approve"
    if pd_value < bands["decline_from"]:
        return "review"
    return "decline"


def create_app(cfg: dict | None = None) -> FastAPI:
    """Build the app. `cfg` defaults to config.yaml; tests pass one with a temp MLflow."""
    cfg = cfg or load_config()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # The champion is loaded ONCE, at startup. Promoting a new champion has no effect on
        # a running server: it is picked up only on restart (a deliberate, visible step).
        try:
            app.state.model, app.state.version = load_champion(cfg)
        except MlflowException as e:
            log.warning("No champion model loaded: %s", e)
            app.state.model, app.state.version = None, None
        yield

    app = FastAPI(title="credit-pd", lifespan=lifespan)

    @app.exception_handler(Exception)
    async def internal_error(request: Request, exc: Exception) -> JSONResponse:
        """Unexpected errors: log the details server-side, return no internals to clients."""
        log.exception("Unhandled error on %s", request.url.path)
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})

    def require_model(request: Request) -> Pipeline:
        """The loaded champion, or 503 if the server started without one."""
        if request.app.state.model is None:
            raise HTTPException(status_code=503, detail="No champion model loaded")
        return request.app.state.model

    @app.get("/health")
    def health(request: Request) -> dict:
        """200 with the serving model version, or 503 if no champion is loaded."""
        require_model(request)
        return {"status": "ok", "model_version": request.app.state.version}

    @app.post("/score")
    def score(applicant: Applicant, request: Request) -> dict:
        """PD for one applicant, rounded to 4 decimals, and its band.

        The band is computed from the rounded PD, so the response is self-consistent (a
        returned pd of 0.15 is always "review").
        """
        model = require_model(request)
        row = pd.DataFrame([applicant.model_dump()], columns=list(FEATURES)).astype(DTYPES)
        pd_value = round(float(model.predict_proba(row)[0, 1]), 4)
        return {
            "pd": pd_value,
            "band": band(pd_value, cfg["bands"]),
            "model_name": cfg["mlflow"]["registered_model_name"],
            "model_version": request.app.state.version,
        }

    return app
