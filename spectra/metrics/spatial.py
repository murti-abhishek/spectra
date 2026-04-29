from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, r2_score


def zonation_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """
    Metrics for continuous zone score regression (0=periportal → 1=pericentral).

    Spearman correlation is primary — it measures whether embeddings encode the
    portocentral gradient, and is robust to non-linear scaling of the score.
    """
    spearman_r, spearman_p = spearmanr(y_true, y_pred)
    pearson_r, pearson_p = pearsonr(y_true, y_pred)
    return {
        "spearman_r": float(spearman_r),
        "spearman_p": float(spearman_p),
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def morans_i(
    values: np.ndarray,
    spatial_coords: np.ndarray,
    n_neighbors: int = 6,
) -> float:
    """
    Moran's I spatial autocorrelation for a scalar field on spatial data.

    Stub — requires squidpy or libpysal (install with pip install spectra[spatial]).
    Raises ImportError with a clear message if the spatial extra is not installed.
    """
    try:
        import libpysal
        from esda.moran import Moran
    except ImportError:
        raise ImportError(
            "morans_i requires the spatial extra: pip install 'spectra[spatial]'"
        ) from None

    from libpysal.weights import KNN
    w = KNN.from_array(spatial_coords, k=n_neighbors)
    w.transform = "r"
    mi = Moran(values, w)
    return float(mi.I)
