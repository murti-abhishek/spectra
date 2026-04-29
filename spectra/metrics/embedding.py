from __future__ import annotations

import numpy as np


def nmi(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """Normalised Mutual Information for clustering evaluation."""
    from sklearn.metrics import normalized_mutual_info_score
    return float(normalized_mutual_info_score(labels_true, labels_pred))


def ari(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """Adjusted Rand Index for clustering evaluation."""
    from sklearn.metrics import adjusted_rand_score
    return float(adjusted_rand_score(labels_true, labels_pred))


def ilisi(
    embeddings: np.ndarray,
    batch_labels: np.ndarray,
    perplexity: int = 30,
) -> float:
    """
    iLISI (integration LISI) — measures batch mixing in embedding space.

    Stub — requires scib (pip install scib-metrics).
    """
    try:
        from scib_metrics.benchmark import Benchmarker  # noqa: F401
    except ImportError:
        raise ImportError(
            "ilisi requires scib-metrics: pip install scib-metrics"
        ) from None
    raise NotImplementedError("ilisi: full implementation deferred to Phase 2 (batch integration task).")
