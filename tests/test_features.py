"""Unit tests for features.py: column selection and robustness to unseen categories."""

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from credit_pd.features import NON_FEATURES, build_preprocessor


def test_preprocessor_excludes_target_and_split(sample_df):
    df = sample_df.assign(split="train")
    columns = [c for _, _, cols in build_preprocessor(df).transformers for c in cols]
    assert not set(NON_FEATURES) & set(columns)
    assert len(columns) == df.shape[1] - len(NON_FEATURES)  # every feature is used


def test_unseen_category_scores_without_error(sample_df):
    X, y = sample_df.drop(columns="default"), sample_df["default"]
    pipe = Pipeline([("prep", build_preprocessor(sample_df)), ("model", LogisticRegression())])
    pipe.fit(X, y)
    p = pipe.predict_proba(X.head(1).assign(purpose="spaceship"))[:, 1]
    assert 0 <= p[0] <= 1
