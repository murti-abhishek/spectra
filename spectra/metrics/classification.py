from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: Optional[list[str]] = None,
) -> dict[str, float]:
    """
    Standard classification metrics for cell type tasks.

    macro_f1 is the primary metric — accuracy is misleading when one class
    (e.g. hepatocytes) dominates cell composition.
    Per-class F1 keys are sanitised: spaces → underscores, lowercased.
    """
    labels = label_names or sorted(np.unique(y_true).tolist())
    per_class = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)

    metrics: dict[str, float] = {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
    }
    for label, score in zip(labels, per_class):
        key = f"f1_{str(label).replace(' ', '_').lower()}"
        metrics[key] = float(score)
    return metrics
