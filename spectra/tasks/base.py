from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
from anndata import AnnData
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import LabelEncoder


class EvalMode(str, Enum):
    ZERO_SHOT = "zero_shot"
    FEW_SHOT = "few_shot"
    FINE_TUNED = "fine_tuned"


@dataclass
class TaskSplit:
    fold: int
    adata_train: AnnData
    adata_test: AnnData
    label_key: str


@dataclass
class TaskResult:
    model: str
    task: str
    fold: int
    mode: EvalMode
    metrics: dict[str, float]
    predictions: Optional[np.ndarray] = None
    labels: Optional[np.ndarray] = None

    def summary(self) -> str:
        metric_str = ", ".join(f"{k}={v:.4f}" for k, v in self.metrics.items())
        return f"[{self.model} | {self.task} | fold={self.fold} | {self.mode}] {metric_str}"


class Task(ABC):
    """Defines a benchmark task: data splits, probe training, and evaluation."""

    name: str
    tissue_specific: bool
    target_tissue: Optional[str]
    requires_spatial: bool
    supported_modes: list[EvalMode]

    @abstractmethod
    def get_splits(self, n_folds: int = 5, seed: int = 42) -> list[TaskSplit]: ...

    @abstractmethod
    def evaluate(
        self,
        predictions: np.ndarray,
        ground_truth: np.ndarray,
        label_names: Optional[list[str]] = None,
    ) -> dict[str, float]: ...

    def get_probe_targets(self, adata: AnnData, label_key: str) -> np.ndarray:
        """Return labels or scores for probe training/evaluation.

        Override in regression tasks to return continuous scores instead of
        categorical labels. The runner calls this on both train and test splits.
        """
        return adata.obs[label_key].values

    def run_linear_probe(
        self,
        embeddings_train: np.ndarray,
        labels_train: np.ndarray,
        embeddings_test: np.ndarray,
        seed: int = 42,
    ) -> np.ndarray:
        """
        Fit logistic regression on embeddings and return test predictions.

        Uses balanced class weights — required for rare liver cell types (Kupffer
        cells, activated HSCs) that are underrepresented but diagnostically important.
        Subclasses override this for regression tasks (zonation score).
        """
        clf = LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=seed,
        )
        clf.fit(embeddings_train, labels_train)
        return clf.predict(embeddings_test)

    def run_regression_probe(
        self,
        embeddings_train: np.ndarray,
        scores_train: np.ndarray,
        embeddings_test: np.ndarray,
        seed: int = 42,
    ) -> np.ndarray:
        """Fit linear regression and return continuous test predictions."""
        reg = LinearRegression()
        reg.fit(embeddings_train, scores_train)
        return reg.predict(embeddings_test)
