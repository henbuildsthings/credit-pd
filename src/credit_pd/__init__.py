"""credit_pd: credit default (PD) scoring service on OpenML credit-g.

Modules:
    config    load `config.yaml` and resolve project paths
    data      fetch, validate, version and split the data (`make ingest`)
    features  preprocessing step of every model Pipeline
    metrics   PD evaluation: AUC, KS, Brier, calibration
    train     fit candidates, pick on val, test once, register in MLflow (`make train`)
    registry  MLflow setup, list/promote versions, load the champion
    api       FastAPI scoring service: PD + decision band (`make serve`)
"""

from importlib.metadata import version

# Read from the installed package metadata, so the version is only defined in pyproject.toml.
__version__ = version("credit-pd")
