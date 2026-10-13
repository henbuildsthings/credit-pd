"""Project configuration: load `config.yaml` and resolve paths from it.

All tunable values (seed, data id, split fractions, file paths, later decision bands) live in
`config.yaml` at the repo root, never hardcoded in the code. Paths in the config are relative
to the repo root, so commands work the same no matter which directory they're run from.
"""

from pathlib import Path

import yaml

# Repo root: this file is src/credit_pd/config.py, so the root is two levels above its folder.
ROOT = Path(__file__).resolve().parents[2]


def load_config(path: Path = ROOT / "config.yaml") -> dict:
    """Read the YAML config into a plain dict (e.g. `cfg["split"]["fractions"]`)."""
    with open(path) as f:
        return yaml.safe_load(f)


def path(cfg: dict, key: str) -> Path:
    """Absolute path for an entry under `paths:` in the config, e.g. `path(cfg, "raw")`."""
    return ROOT / cfg["paths"][key]
