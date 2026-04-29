from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import scipy.sparse as sp
from anndata import AnnData
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LogisticRegression

from spectra.models.base import InputType, ModelAdapter, ModelInfo

logger = logging.getLogger(__name__)


class PCABaselineAdapter(ModelAdapter):
    """
    PCA + logistic/linear regression baseline.

    Recent benchmarks (Hou et al. 2026) show this is competitive with or beats
    scFMs on many tasks. LiverTransformer must be compared against this baseline.

    State management:
        get_embeddings() fits PCA on the first call, transforms on subsequent calls.
        Call reset() between folds to clear fitted state.
    """

    def __init__(self, n_components: int = 50, seed: int = 42):
        self._n_components = n_components
        self._seed = seed
        self._pca: Optional[PCA] = None
        self._head = None  # logistic or linear regression head

    @property
    def info(self) -> ModelInfo:
        return ModelInfo(
            name="pca_baseline",
            input_type=InputType.RAW,
            embedding_dim=self._n_components,
            supports_fine_tuning=True,
        )

    def get_embeddings(self, adata: AnnData, layer: Optional[str] = None) -> np.ndarray:
        X = self._to_dense(adata, layer)
        if self._pca is None:
            n_components = min(self._n_components, X.shape[0] - 1, X.shape[1])
            self._pca = PCA(n_components=n_components, random_state=self._seed)
            logger.info("PCABaseline: fitting PCA (n_components=%d) on %d cells.", n_components, X.shape[0])
            return self._pca.fit_transform(X).astype(np.float32)
        return self._pca.transform(X).astype(np.float32)

    def fine_tune(
        self,
        adata_train: AnnData,
        task_type: str,
        label_key: str,
        n_epochs: int = 10,
        **kwargs,
    ) -> None:
        embeddings = self.get_embeddings(adata_train)
        y = adata_train.obs[label_key].values
        if task_type.endswith("_regression"):
            self._head = LinearRegression()
        else:
            self._head = LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=self._seed,
            )
        self._head.fit(embeddings, y)

    def predict(self, adata: AnnData, task_type: str) -> np.ndarray:
        if self._head is None:
            raise RuntimeError("Call fine_tune() before predict().")
        embeddings = self.get_embeddings(adata)
        return self._head.predict(embeddings)

    def reset(self) -> None:
        """Clear fitted PCA and head. Must be called between CV folds."""
        self._pca = None
        self._head = None

    def _to_dense(self, adata: AnnData, layer: Optional[str]) -> np.ndarray:
        X = adata.layers[layer] if layer is not None else adata.X
        if sp.issparse(X):
            X = X.toarray()
        return X.astype(np.float32)
