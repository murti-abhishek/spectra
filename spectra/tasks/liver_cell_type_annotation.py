from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from anndata import AnnData
from sklearn.model_selection import StratifiedKFold

from spectra.metrics.classification import classification_metrics
from spectra.tasks.base import EvalMode, Task, TaskSplit

logger = logging.getLogger(__name__)


class LiverCellTypeAnnotationTask(Task):
    """
    Liver cell type annotation at coarse or fine resolution.

    Coarse: broad cell type categories (Hepatocytes, Kupffer cells, LSECs, …).
    Fine: adds zone-specific hepatocyte subtypes (periportal / mid / pericentral).

    Macro-F1 is the primary metric — accuracy is misleading because hepatocytes
    dominate the composition and would inflate scores on a majority-class predictor.
    """

    name = "liver_cell_type_annotation"
    tissue_specific = True
    target_tissue = "liver"
    requires_spatial = False
    supported_modes = [EvalMode.ZERO_SHOT, EvalMode.FINE_TUNED]

    # Reference label sets — used for validation warnings, not hard enforcement.
    COARSE_CELL_TYPES = frozenset([
        "Hepatocytes",
        "Kupffer cells",
        "Hepatic stellate cells",
        "LSECs",
        "Cholangiocytes",
        "Portal fibroblasts",
    ])
    FINE_CELL_TYPES = frozenset([
        "Hepatocytes_periportal",
        "Hepatocytes_mid",
        "Hepatocytes_pericentral",
        "Kupffer cells",
        "Hepatic stellate cells quiescent",
        "Hepatic stellate cells activated",
        "LSECs",
        "Cholangiocytes",
        "Portal fibroblasts",
    ])

    def __init__(
        self,
        adata: AnnData,
        label_key: str,
        resolution: str = "coarse",
    ):
        if resolution not in ("coarse", "fine"):
            raise ValueError(f"resolution must be 'coarse' or 'fine', got {resolution!r}")
        if label_key not in adata.obs.columns:
            raise KeyError(f"label_key {label_key!r} not found in adata.obs")

        self._adata = adata
        self._label_key = label_key
        self._resolution = resolution

        observed = set(adata.obs[label_key].unique())
        expected = self.COARSE_CELL_TYPES if resolution == "coarse" else self.FINE_CELL_TYPES
        unexpected = observed - expected
        if unexpected:
            logger.warning(
                "LiverCellTypeAnnotation (%s): unexpected labels in data: %s. "
                "Proceeding — ensure these are intentional.",
                resolution, sorted(unexpected),
            )

    def get_splits(self, n_folds: int = 5, seed: int = 42) -> list[TaskSplit]:
        labels = self._adata.obs[self._label_key].values
        counts = {c: (labels == c).sum() for c in np.unique(labels)}
        rare = {c: n for c, n in counts.items() if n < n_folds}
        if rare:
            logger.warning(
                "LiverCellTypeAnnotation: %d class(es) have fewer samples than n_folds=%d: %s. "
                "Some folds may lack these classes.",
                len(rare), n_folds, rare,
            )

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

    def evaluate(
        self,
        predictions: np.ndarray,
        ground_truth: np.ndarray,
        label_names: Optional[list[str]] = None,
    ) -> dict[str, float]:
        return classification_metrics(ground_truth, predictions, label_names)
