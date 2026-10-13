"""credit_pd: credit default (PD) scoring service on OpenML credit-g.

Modules:
    config  load `config.yaml` and resolve project paths
    data    fetch, validate, version and split the data (`make ingest`)
"""

from importlib.metadata import version

# Read from the installed package metadata, so the version is only defined in pyproject.toml.
__version__ = version("credit-pd")
