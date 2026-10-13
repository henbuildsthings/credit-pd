"""Evaluation metrics for a PD model: ranking, calibration and overall level.

A PD model has two jobs: rank customers by risk (AUC, KS) and give probabilities that
match reality, because they feed pricing, decision bands and expected loss (Brier score,
mean prediction vs observed rate, calibration table).
"""

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve


def calibration_table(y_true, p, n_bins: int = 10) -> pd.DataFrame:
    """Group rows into equal-sized bins by predicted probability (deciles by default).

    For each bin: row count, mean predicted PD and observed default rate. A calibrated
    model has mean_pred close to observed_rate in every bin. Bins are cut on the rank of
    `p`, so ties (e.g. a constant baseline) still give equal-sized bins instead of an error.
    """
    df = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(p)})
    df["bin"] = pd.qcut(df["p"].rank(method="first"), n_bins, labels=False) + 1
    stats = {"n": ("y", "size"), "mean_pred": ("p", "mean"), "observed_rate": ("y", "mean")}
    return df.groupby("bin").agg(**stats)


def evaluate(y_true, p) -> dict:
    """Score predicted PDs `p` against outcomes `y_true` (1 = default). JSON-ready dict.

    - auc: probability a random defaulter gets a higher PD than a random non-defaulter
      (0.5 = random, 1 = perfect). Ranking only; ignores whether the levels are right.
    - ks: largest gap between the cumulative score distributions of defaulters and
      non-defaulters = max(TPR - FPR). A classic credit-scoring ranking metric.
    - brier: mean squared error of the probabilities. Penalises bad ranking and bad
      calibration alike; lower is better.
    - mean_pred vs observed_rate: is the overall PD level right?
    - calibration: per-decile table (see `calibration_table`).
    """
    fpr, tpr, _ = roc_curve(y_true, p)
    return {
        "auc": float(roc_auc_score(y_true, p)),
        "ks": float(np.max(tpr - fpr)),
        "brier": float(brier_score_loss(y_true, p)),
        "mean_pred": float(np.mean(p)),
        "observed_rate": float(np.mean(y_true)),
        "calibration": calibration_table(y_true, p).reset_index().to_dict("records"),
    }
