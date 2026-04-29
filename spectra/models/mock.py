from __future__ import annotations

from typing import Optional

import numpy as np
from anndata import AnnData

from spectra.models.base import InputType, ModelAdapter, ModelInfo


class MockModelAdapter(ModelAdapter):
    """Returns random Gaussian embeddings. For testing only — never use in real runs."""

    def __init__(self, embedding_dim: int = 64, seed: int = 42):
        self._embedding_dim = embedding_dim
        self._rng = np.random.default_rng(seed)
        self._classes: Optional[np.ndarray] = None

    @property
    def info(self) -> ModelInfo:
        return ModelInfo(
            name="mock",
            input_type=InputType.RAW,
            embedding_dim=self._embedding_dim,
            supports_fine_tuning=True,
        )

    def get_embeddings(self, adata: AnnData, layer: Optional[str] = None) -> np.ndarray:
        return self._rng.standard_normal((adata.n_obs, self._embedding_dim)).astype(np.float32)

    def fine_tune(
        self,
        adata_train: AnnData,
        task_type: str,
        label_key: str,
        n_epochs: int = 10,
        **kwargs,
    ) -> None:
        self._classes = np.unique(adata_train.obs[label_key].values)

    def predict(self, adata: AnnData, task_type: str) -> np.ndarray:
        if task_type.endswith("_regression"):
            return self._rng.uniform(0.0, 1.0, size=adata.n_obs).astype(np.float32)
        if self._classes is not None:
            indices = self._rng.integers(0, len(self._classes), size=adata.n_obs)
            return self._classes[indices]
        return self._rng.uniform(0.0, 1.0, size=adata.n_obs).astype(np.float32)

    def reset(self) -> None:
        self._classes = None
