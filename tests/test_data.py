"""Unit tests for data.py: schema, content hash and split. No network, no data/ folder."""

import pandas as pd
import pytest

from credit_pd.data import content_sha256, split, validate


def test_schema_accepts_sample(sample_df):
    assert validate(sample_df).equals(sample_df)


@pytest.mark.parametrize(
    "corrupt",
    [
        pytest.param(lambda df: df.assign(extra=1), id="extra-column"),
        pytest.param(lambda df: df.drop(columns="age"), id="missing-column"),
        pytest.param(lambda df: df.assign(age=999), id="age-out-of-range"),
        pytest.param(lambda df: df.assign(housing="boat"), id="unknown-category"),
    ],
)
def test_schema_rejects(sample_df, corrupt):
    with pytest.raises(ValueError, match="failed schema validation"):
        validate(corrupt(sample_df))


def test_hash_survives_parquet_round_trip(sample_df, tmp_path):
    sample_df.to_parquet(tmp_path / "d.parquet", index=False)
    assert content_sha256(pd.read_parquet(tmp_path / "d.parquet")) == content_sha256(sample_df)


def test_hash_changes_when_one_value_changes(sample_df):
    changed = sample_df.copy()
    changed.loc[0, "age"] += 1
    assert content_sha256(changed) != content_sha256(sample_df)


def test_split_is_deterministic(sample_df, cfg):
    assert split(sample_df, cfg)["split"].equals(split(sample_df, cfg)["split"])


def test_split_is_stratified(sample_df, cfg):
    """Each split holds its fraction of the defaults, give or take one row of rounding."""
    out = split(sample_df, cfg)
    total_defaults = sample_df["default"].sum()
    for name, frac in cfg["split"]["fractions"].items():
        defaults = out.loc[out["split"] == name, "default"].sum()
        assert abs(defaults - frac * total_defaults) <= 1, name
