"""Data ingestion: fetch, validate, version and split OpenML credit-g.

Run with `make ingest` or `python -m credit_pd.data [--force]`.

Pipeline (see `ingest`):
    1. Load the raw data: from `data/raw/credit-g.parquet` if present, else from OpenML.
    2. Validate it against `SCHEMA`, a strict pandera schema built from `FEATURES`.
    3. Fingerprint it with a content hash (`content_sha256`).
    4. Split it into train/val/test/future, stratified on the target and seeded.
    5. Compare everything with the committed `data/manifest.json`.

The manifest is the data's identity card. It's committed to git while the data itself is
not, so any machine can check it has exactly the same data. Ingest is strict: if anything
differs from the manifest it raises and writes nothing. The only way to change the data is
a deliberate `--force`, which re-downloads and rewrites the manifest (a visible git diff).

All settings (data id, target, split fractions, seed, paths) come from `config.yaml`.
"""

import argparse
import hashlib
import json

import pandas as pd
import pandera.pandas as pa
from pandera.errors import SchemaErrors
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

from credit_pd.config import load_config, path

# The feature contract, in training column order: the single source of truth for what a
# valid applicant looks like. A list = categorical string column with exactly these
# allowed values (as seen in the real data). A dict = integer column with these bounds
# (pandera and pydantic both understand ge/gt/le). Used by the pandera schema below and
# by the API's request model (api.py), so the two can never drift apart.
FEATURES: dict[str, list[str] | dict[str, int]] = {
    "checking_status": ["<0", "0<=X<200", ">=200", "no checking"],
    "duration": {"ge": 1, "le": 120},  # loan duration in months
    "credit_history": [
        "all paid", "critical/other existing credit", "delayed previously",
        "existing paid", "no credits/all paid",
    ],
    "purpose": [
        "business", "domestic appliance", "education", "furniture/equipment", "new car",
        "other", "radio/tv", "repairs", "retraining", "used car",
    ],
    "credit_amount": {"gt": 0, "le": 100_000},  # in Deutsche Mark
    "savings_status": ["<100", "100<=X<500", "500<=X<1000", ">=1000", "no known savings"],
    "employment": ["unemployed", "<1", "1<=X<4", "4<=X<7", ">=7"],
    "installment_commitment": {},
    "personal_status": ["female div/dep/mar", "male div/sep", "male mar/wid", "male single"],
    "other_parties": ["co applicant", "guarantor", "none"],
    "residence_since": {},
    "property_magnitude": ["car", "life insurance", "no known property", "real estate"],
    "age": {"ge": 18, "le": 100},
    "other_payment_plans": ["bank", "none", "stores"],
    "housing": ["for free", "own", "rent"],
    "existing_credits": {},
    "job": [
        "high qualif/self emp/mgmt", "skilled", "unemp/unskilled non res", "unskilled resident",
    ],
    "num_dependents": {},
    "own_telephone": ["none", "yes"],
    "foreign_worker": ["no", "yes"],
}  # fmt: skip

# pandas dtype per feature, as in training: categoricals are plain strings, the rest int64.
DTYPES = {name: "str" if isinstance(spec, list) else "int64" for name, spec in FEATURES.items()}


def _column(spec: list[str] | dict[str, int]) -> pa.Column:
    """pandera column for one FEATURES entry: allowed values for strings, bounds for ints."""
    if isinstance(spec, list):
        return pa.Column(str, pa.Check.isin(spec))
    return pa.Column(int, [getattr(pa.Check, op)(value) for op, value in spec.items()])


# Contract for the credit-g data after `fetch`: the 20 FEATURES + binary `default` target.
# - `strict=True`: a missing or unexpected column fails, so upstream changes can't slip
#   through unnoticed.
# - Categorical columns are plain strings limited to known values: a new category (e.g. a
#   new `purpose`) fails here instead of confusing the model later.
# - Numeric ranges are sanity bounds, not tight statistical limits: they catch broken data
#   (negative amounts, age 150), not unusual but valid customers.
# - Nulls aren't allowed anywhere (pandera's default is nullable=False).
SCHEMA = pa.DataFrameSchema(
    {name: _column(spec) for name, spec in FEATURES.items()}
    | {"default": pa.Column(int, pa.Check.isin([0, 1]))},  # target: 1 = bad credit
    name="credit-g",
    strict=True,
)


def fetch(cfg: dict) -> pd.DataFrame:
    """Download credit-g from OpenML and reshape it into the schema's format.

    - Fetched by pinned `data_id`, never by name: a name can later resolve to a different
      dataset version, an id can't. `cache=False` makes it a real download, so `--force`
      gets fresh data rather than sklearn's local copy.
    - The text target (`class`: good/bad) becomes an int column `default` (bad=1, good=0),
      and `class` is dropped. The target must have exactly two labels, one of them
      `positive_label`; otherwise an unexpected label would silently become 0.
    - pandas `category` columns become plain strings, so the schema and the content hash
      don't depend on how pandas happened to order the categories.
    """
    d = cfg["data"]
    df = fetch_openml(data_id=d["openml_data_id"], as_frame=True, parser="auto", cache=False).frame
    labels = set(df[d["target"]].unique())
    if d["positive_label"] not in labels or len(labels) != 2:
        raise ValueError(f"Expected binary target with {d['positive_label']!r}, got {labels}")
    df["default"] = (df.pop(d["target"]) == d["positive_label"]).astype("int64")
    return df.astype({c: str for c in df.select_dtypes("category").columns})


def validate(df: pd.DataFrame) -> pd.DataFrame:
    """Check `df` against `SCHEMA` and return it unchanged if it passes.

    `lazy=True` collects every failure instead of stopping at the first one. On failure it
    raises a ValueError with a compact table (column, check, failing value, row index),
    because pandera's own message prints the entire schema and buries the actual problem.
    """
    try:
        return SCHEMA.validate(df, lazy=True)
    except SchemaErrors as e:
        cases = e.failure_cases[["column", "check", "failure_case", "index"]].to_string(index=False)
        raise ValueError(f"credit-g failed schema validation:\n{cases}") from e


def content_sha256(df: pd.DataFrame) -> str:
    """Fingerprint the data itself, not the file it's stored in.

    Hashes column names + dtypes, then the values row by row (`hash_pandas_object`, index
    ignored). File bytes would change with the parquet writer version or compression even
    when the data is identical; this hash only changes when the data does. Including dtypes
    catches silent type drift, e.g. an int column turning into float.
    """
    h = hashlib.sha256(json.dumps({c: str(t) for c, t in df.dtypes.items()}).encode())
    h.update(pd.util.hash_pandas_object(df, index=False).to_numpy().tobytes())
    return h.hexdigest()


def split(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Return `df` with a `split` column assigning each row to train/val/test/future.

    The splits are carved off one at a time, in the order they appear in the config, each
    stratified on `default` so every split keeps the same default rate. Sizes are computed
    as whole row counts up front (e.g. 0.15 * 1000 = 150), so float rounding can't make a
    split one row short; the last split gets whatever is left. The same seed always gives
    the same split. "future" is never used for training: it stands in for later data.
    """
    fracs = cfg["split"]["fractions"]
    if abs(sum(fracs.values()) - 1) > 1e-9:
        raise ValueError(f"split fractions must sum to 1, got {fracs}")
    out, rest = df.assign(split=""), df.index  # `rest` = rows not yet assigned
    for name in list(fracs)[:-1]:
        n = round(fracs[name] * len(df))
        take, rest = train_test_split(
            rest, train_size=n, stratify=df.loc[rest, "default"], random_state=cfg["seed"]
        )
        out.loc[take, "split"] = name
    out.loc[rest, "split"] = list(fracs)[-1]
    return out


def build_manifest(df: pd.DataFrame, splits: pd.DataFrame, cfg: dict) -> dict:
    """Describe the data and its splits; this dict is what `data/manifest.json` stores.

    Besides the content hash it records the source, shape, columns, seed, and the row count
    and default rate of each split, so a changed seed or split config is caught too.
    """
    stats = splits.groupby("split")["default"].agg(n_rows="size", default_rate="mean").round(4)
    return {
        "source": "openml",
        "data_id": cfg["data"]["openml_data_id"],
        "n_rows": len(df),
        "n_cols": df.shape[1],
        "columns": list(df.columns),
        "content_sha256": content_sha256(df),
        "seed": cfg["seed"],
        "splits": stats.loc[list(cfg["split"]["fractions"])].to_dict("index"),  # config order
    }


def ingest(force: bool = False) -> None:
    """Run the full ingest and enforce the manifest. Safe to run repeatedly.

    Behaviour:
    - raw file present, matches manifest -> no download, prints "up to date"
    - raw file missing                   -> download; it must still match the manifest
    - no manifest yet (first run)        -> write it
    - any mismatch                       -> raise RuntimeError, nothing is written
    - force=True                         -> re-download and rewrite the manifest

    Everything is computed and checked in memory before any file is written, so a failed
    run never leaves a half-updated raw file or manifest behind. The splits file is always
    rewritten; it's deterministic, so its content doesn't change.
    """
    cfg = load_config()
    raw, manifest_path = path(cfg, "raw"), path(cfg, "manifest")
    # With --force the old manifest is ignored, so nothing can mismatch and it's rewritten.
    old = json.loads(manifest_path.read_text()) if manifest_path.exists() and not force else None
    downloaded = force or not raw.exists()
    df = validate(fetch(cfg) if downloaded else pd.read_parquet(raw))
    splits = split(df, cfg)
    manifest = build_manifest(df, splits, cfg)

    if old is not None and old != manifest:
        diff = [k for k in manifest if old.get(k) != manifest[k]]
        msg = f"Data differs from manifest in {diff}; nothing written. Use --force if deliberate."
        raise RuntimeError(msg)

    # All checks passed: now it's safe to write.
    if downloaded:
        raw.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(raw, index=False)
    splits_path = path(cfg, "splits")
    splits_path.parent.mkdir(parents=True, exist_ok=True)
    splits.to_parquet(splits_path, index=False)
    if old is None:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    status = "downloaded" if downloaded else "up to date"
    print(f"{status}: sha256={manifest['content_sha256'][:12]} manifest_written={old is None}")
    print(pd.DataFrame.from_dict(manifest["splits"], orient="index").to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="re-download and rewrite manifest")
    ingest(force=parser.parse_args().force)
