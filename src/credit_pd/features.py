"""Feature preprocessing: the first step of every model Pipeline.

The preprocessor is fitted inside the sklearn Pipeline, together with the model, so the
deployable unit is one object that takes raw credit-g rows and returns probabilities. The
API never has to repeat any preprocessing, and train/serve skew can't creep in.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Columns that are in the splits file but are not model inputs.
NON_FEATURES = ["default", "split"]


def build_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    """Return an unfitted ColumnTransformer for the feature columns of `df`.

    Column lists come from the dtypes (numbers vs strings), not a hardcoded list, so the
    schema in `data.py` stays the single place that defines the columns.
    - Numeric -> StandardScaler: puts features on one scale, which logistic regression
      needs for its regularisation to treat them fairly (trees don't care either way).
    - Categorical -> one-hot. `handle_unknown="ignore"` turns a category never seen in
      training into all zeros instead of an error at scoring time. Dense output keeps
      the pipeline simple; with ~60 columns and 1000 rows memory is no concern.
    """
    features = df.drop(columns=NON_FEATURES, errors="ignore")
    numeric = list(features.select_dtypes("number").columns)
    categorical = list(features.select_dtypes(["object", "string"]).columns)
    return ColumnTransformer(
        [
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
        ]
    )
