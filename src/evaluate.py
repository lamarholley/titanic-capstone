"""Model evaluation metrics, shared by training and by the test suite."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(y_true, y_pred, y_proba=None) -> dict:
    """Compute the standard classification metrics used throughout this
    project. `y_proba` is the predicted probability of the positive class
    (Survived == 1); if omitted, ROC-AUC is skipped.
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_proba is not None:
        # roc_auc_score needs both classes present to be defined.
        if len(np.unique(y_true)) == 2:
            metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
    return metrics
