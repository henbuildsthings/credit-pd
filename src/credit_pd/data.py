"""Fetch, validate, version and split OpenML credit-g. Run: python -m credit_pd.data [--force]."""

import argparse
import hashlib
import json

import pandas as pd
import pandera.pandas as pa
from pandera.errors import SchemaErrors
from pandera.typing import Series
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

from credit_pd.config import load_config, path

Str = Series[str]


def _cat(*values: str):
    return pa.Field(isin=list(values))


class CreditSchema(pa.DataFrameModel):
    checking_status: Str = _cat("<0", "0<=X<200", ">=200", "no checking")
    duration: Series[int] = pa.Field(ge=1, le=120)
    credit_history: Str = _cat(
        "all paid", "critical/other existing credit", "delayed previously",
        "existing paid", "no credits/all paid",
    )  # fmt: skip
    purpose: Str = _cat(
        "business", "domestic appliance", "education", "furniture/equipment", "new car",
        "other", "radio/tv", "repairs", "retraining", "used car",
    )  # fmt: skip
    credit_amount: Series[int] = pa.Field(gt=0, le=100_000)
    savings_status: Str = _cat("<100", "100<=X<500", "500<=X<1000", ">=1000", "no known savings")
    employment: Str = _cat("unemployed", "<1", "1<=X<4", "4<=X<7", ">=7")
    installment_commitment: Series[int]
    personal_status: Str = _cat("female div/dep/mar", "male div/sep", "male mar/wid", "male single")
    other_parties: Str = _cat("co applicant", "guarantor", "none")
    residence_since: Series[int]
    property_magnitude: Str = _cat("car", "life insurance", "no known property", "real estate")
    age: Series[int] = pa.Field(ge=18, le=100)
    other_payment_plans: Str = _cat("bank", "none", "stores")
    housing: Str = _cat("for free", "own", "rent")
    existing_credits: Series[int]
    job: Str = _cat(
        "high qualif/self emp/mgmt", "skilled", "unemp/unskilled non res", "unskilled resident"
    )
    num_dependents: Series[int]
    own_telephone: Str = _cat("none", "yes")
    foreign_worker: Str = _cat("no", "yes")
    default: Series[int] = pa.Field(isin=[0, 1])

    class Config:
        strict = True  # missing or unexpected columns fail


def fetch(cfg: dict) -> pd.DataFrame:
    d = cfg["data"]
    df = fetch_openml(data_id=d["openml_data_id"], as_frame=True, parser="auto", cache=False).frame
    labels = set(df[d["target"]].unique())
    if d["positive_label"] not in labels or len(labels) != 2:
        raise ValueError(f"Expected binary target with {d['positive_label']!r}, got {labels}")
    df["default"] = (df.pop(d["target"]) == d["positive_label"]).astype("int64")
    return df.astype({c: str for c in df.select_dtypes("category").columns})


def validate(df: pd.DataFrame) -> pd.DataFrame:
    try:
        return CreditSchema.validate(df, lazy=True)
    except SchemaErrors as e:
        cases = e.failure_cases[["column", "check", "failure_case", "index"]].to_string(index=False)
        raise ValueError(f"credit-g failed schema validation:\n{cases}") from e


def content_sha256(df: pd.DataFrame) -> str:
    h = hashlib.sha256(json.dumps({c: str(t) for c, t in df.dtypes.items()}).encode())
    h.update(pd.util.hash_pandas_object(df, index=False).to_numpy().tobytes())
    return h.hexdigest()


def split(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    fracs = cfg["split"]["fractions"]
    if abs(sum(fracs.values()) - 1) > 1e-9:
        raise ValueError(f"split fractions must sum to 1, got {fracs}")
    out, rest = df.assign(split=""), df.index
    for name in list(fracs)[:-1]:
        n = round(fracs[name] * len(df))
        take, rest = train_test_split(
            rest, train_size=n, stratify=df.loc[rest, "default"], random_state=cfg["seed"]
        )
        out.loc[take, "split"] = name
    out.loc[rest, "split"] = list(fracs)[-1]
    return out


def build_manifest(df: pd.DataFrame, splits: pd.DataFrame, cfg: dict) -> dict:
    stats = splits.groupby("split")["default"].agg(n_rows="size", default_rate="mean").round(4)
    return {
        "source": "openml",
        "data_id": cfg["data"]["openml_data_id"],
        "n_rows": len(df),
        "n_cols": df.shape[1],
        "columns": list(df.columns),
        "content_sha256": content_sha256(df),
        "seed": cfg["seed"],
        "splits": stats.loc[list(cfg["split"]["fractions"])].to_dict("index"),
    }


def ingest(force: bool = False) -> None:
    cfg = load_config()
    raw, manifest_path = path(cfg, "raw"), path(cfg, "manifest")
    old = json.loads(manifest_path.read_text()) if manifest_path.exists() and not force else None
    downloaded = force or not raw.exists()
    df = validate(fetch(cfg) if downloaded else pd.read_parquet(raw))
    splits = split(df, cfg)
    manifest = build_manifest(df, splits, cfg)
    if old is not None and old != manifest:
        diff = [k for k in manifest if old.get(k) != manifest[k]]
        msg = f"Data differs from manifest in {diff}; nothing written. Use --force if deliberate."
        raise RuntimeError(msg)
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-download and rewrite manifest")
    ingest(force=parser.parse_args().force)
