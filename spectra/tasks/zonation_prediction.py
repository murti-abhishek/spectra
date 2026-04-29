from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from anndata import AnnData
from sklearn.model_selection import KFold, StratifiedKFold

from spectra.metrics.classification import classification_metrics
from spectra.metrics.spatial import zonation_regression_metrics
from spectra.tasks.base import EvalMode, Task, TaskSplit

logger = logging.getLogger(__name__)

_ZONE_LABELS = ["periportal", "mid", "pericentral"]
# Ordered 0 → 1 along the portocentral axis.
_ZONE_TO_SCORE: dict[str, float] = {"periportal": 0.0, "mid": 0.5, "pericentral": 1.0}


class ZonationPredictionTask(Task):
    """
    Hepatocyte zonation prediction in two formulations:

    classification — periportal / mid / pericentral (3-class, macro-F1 primary).
    regression     — continuous zone score 0 (periportal) → 1 (pericentral)
                     (Spearman correlation primary).

    Regression is the harder, more informative test: it checks whether embeddings
    encode the portocentral *gradient*, not just the extremes.

    If mode='regression' and score_key is None, the continuous score is derived
    from label_key using the mapping periportal=0, mid=0.5, pericentral=1.
    Real MERFISH data should supply a distance-based score via score_key instead.
    """

    name = "zonation_prediction"
    tissue_specific = True
    target_tissue = "liver"
    requires_spatial = False
    supported_modes = [EvalMode.ZERO_SHOT, EvalMode.FINE_TUNED]

    def __init__(
        self,
        adata: AnnData,
        label_key: str,
        mode: str = "classification",
        score_key: Optional[str] = None,
    ):
        if mode not in ("classification", "regression"):
            raise ValueError(f"mode must be 'classification' or 'regression', got {mode!r}")
        if label_key not in adata.obs.columns:
            raise KeyError(f"label_key {label_key!r} not found in adata.obs")
        if score_key is not None and score_key not in adata.obs.columns:
            raise KeyError(f"score_key {score_key!r} not found in adata.obs")

        self._adata = adata
        self._label_key = label_key
        self._mode = mode
        self._score_key = score_key

        if mode == "regression" and score_key is None:
            logger.warning(
                "ZonationPrediction: score_key not provided. Deriving continuous zone score "
                "from label_key using periportal=0 / mid=0.5 / pericentral=1. "
                "For MERFISH data, supply a distance-based score via score_key instead."
            )

    # ------------------------------------------------------------------
    # Task interface
    # ------------------------------------------------------------------

    def get_splits(self, n_folds: int = 5, seed: int = 42) -> list[TaskSplit]:
        if self._mode == "classification":
            return self._stratified_splits(n_folds, seed)
        return self._kfold_splits(n_folds, seed)

    def evaluate(
        self,
        predictions: np.ndarray,
        ground_truth: np.ndarray,
        label_names: Optional[list[str]] = None,
    ) -> dict[str, float]:
        if self._mode == "regression":
            return self._evaluate_regression(predictions, ground_truth)
        return self._evaluate_classification(predictions, ground_truth, label_names)

    def run_linear_probe(
        self,
        embeddings_train: np.ndarray,
        labels_train: np.ndarray,
        embeddings_test: np.ndarray,
        seed: int = 42,
    ) -> np.ndarray:
        if self._mode == "regression":
            return self.run_regression_probe(embeddings_train, labels_train, embeddings_test)
        return super().run_linear_probe(embeddings_train, labels_train, embeddings_test, seed)

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def get_zone_scores(self, adata: Optional[AnnData] = None) -> np.ndarray:
        """Return continuous zone scores for adata (or self._adata if None)."""
        src = adata if adata is not None else self._adata
        if self._score_key is not None:
            return src.obs[self._score_key].values.astype(np.float32)
        return self.derive_zone_score(src.obs[self._label_key].values)

    @staticmethod
    def derive_zone_score(labels: np.ndarray) -> np.ndarray:
        """Map categorical zone labels to continuous 0-1 scores."""
        scores = np.array([_ZONE_TO_SCORE[str(l).lower()] for l in labels], dtype=np.float32)
        return scores

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _stratified_splits(self, n_folds: int, seed: int) -> list[TaskSplit]:
        labels = self._adata.obs[self._label_key].values
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        splits = []
        for fold, (train_idx, test_idx) in enumerate(
            skf.split(np.zeros(len(labels)), labels)
        ):
            splits.append(TaskSplit(
                fold=fold,
                adata_train=self._adata[train_idx].copy(),
                adata_test=self._adata[test_idx].copy(),
                label_key=self._label_key,
            ))
        return splits

    def _kfold_splits(self, n_folds: int, seed: int) -> list[TaskSplit]:
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        n = self._adata.n_obs
        splits = []
        for fold, (train_idx, test_idx) in enumerate(kf.split(np.zeros(n))):
            splits.append(TaskSplit(
                fold=fold,
                adata_train=self._adata[train_idx].copy(),
                adata_test=self._adata[test_idx].copy(),
                label_key=self._label_key,
            ))
        return splits

    def get_probe_targets(self, adata: AnnData, label_key: str) -> np.ndarray:
        if self._mode == "regression":
            return self.get_zone_scores(adata)
        return adata.obs[label_key].values

    def _evaluate_regression(
        self, predictions: np.ndarray, ground_truth: np.ndarray
    ) -> dict[str, float]:
        return zonation_regression_metrics(ground_truth, predictions)

    def _evaluate_classification(
        self,
        predictions: np.ndarray,
        ground_truth: np.ndarray,
        label_names: Optional[list[str]],
    ) -> dict[str, float]:
        return classification_metrics(ground_truth, predictions, label_names or _ZONE_LABELS)
