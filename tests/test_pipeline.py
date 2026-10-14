"""Smoke test: the whole training entry point runs and registers a version."""

from credit_pd.data import split
from credit_pd.registry import list_versions
from credit_pd.train import train


def test_train_end_to_end_registers_a_version(tmp_mlflow, sample_df):
    version = train(tmp_mlflow, split(sample_df, tmp_mlflow))
    assert version == 1
    assert list(list_versions(tmp_mlflow).index) == [1]
