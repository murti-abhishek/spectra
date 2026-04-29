from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
from anndata import AnnData

logger = logging.getLogger(__name__)


class InputType(str, Enum):
    RANK = "rank"        # Geneformer: rank-ordered gene indices
    BINNED = "binned"    # scGPT: expression binned into discrete levels
    RAW = "raw"          # scFoundation, LiverTransformer: raw counts
    UNIVERSAL = "universal"  # UCE: protein-level embeddings, vocabulary-agnostic


@dataclass
class ModelInfo:
    name: str
    input_type: InputType
    pretraining_cells: Optional[int] = None
    # None for universal embedders (UCE) that are vocabulary-agnostic
    gene_vocabulary: Optional[list[str]] = None
    embedding_dim: Optional[int] = None
    supports_fine_tuning: bool = True
    requires_spatial: bool = False
    notes: str = ""


class ModelAdapter(ABC):
    """Standard interface wrapping a single-cell foundation model."""

    @property
    @abstractmethod
    def info(self) -> ModelInfo: ...

    @abstractmethod
    def get_embeddings(self, adata: AnnData, layer: Optional[str] = None) -> np.ndarray:
        """Return (n_cells, d_model) embedding matrix. Gene harmonization happens here."""

    @abstractmethod
    def fine_tune(
        self,
        adata_train: AnnData,
        task_type: str,
        label_key: str,
        n_epochs: int = 10,
        **kwargs,
    ) -> None:
        """Fine-tune on labelled data. task_type is a string like 'zonation_regression'."""

    @abstractmethod
    def predict(self, adata: AnnData, task_type: str) -> np.ndarray:
        """Predictions after fine-tuning. Returns class indices or continuous scores."""

    def reset(self) -> None:
        """Clear any fitted state. Called by BenchmarkRunner between CV folds."""

    def _harmonize_genes(
        self,
        adata: AnnData,
        min_overlap: float = 0.05,
    ) -> AnnData:
        """
        Map adata var_names to this model's gene vocabulary.

        Fills missing genes with 0. Raises if overlap < min_overlap (producing noise).
        Warns between min_overlap and 0.5; logs info between 0.5 and 0.8.
        Returns a new AnnData aligned to the model vocabulary.
        """
        vocab = self.info.gene_vocabulary
        if vocab is None:
            # Universal embedder — no harmonization needed
            return adata

        adata_genes = set(adata.var_names)
        vocab_set = set(vocab)
        overlap = adata_genes & vocab_set
        overlap_frac = len(overlap) / len(vocab_set)

        if overlap_frac < min_overlap:
            raise ValueError(
                f"{self.info.name}: gene overlap is {overlap_frac:.1%} "
                f"({len(overlap)}/{len(vocab_set)} vocabulary genes present). "
                f"Minimum required: {min_overlap:.0%}. "
                "Check that gene identifiers match (Ensembl IDs vs symbols)."
            )
        elif overlap_frac < 0.5:
            logger.warning(
                "%s: low gene overlap %.1f%% (%d/%d vocabulary genes). "
                "Embeddings may be unreliable.",
                self.info.name, overlap_frac * 100, len(overlap), len(vocab_set),
            )
        elif overlap_frac < 0.8:
            logger.info(
                "%s: moderate gene overlap %.1f%% (%d/%d vocabulary genes).",
                self.info.name, overlap_frac * 100, len(overlap), len(vocab_set),
            )

        import scipy.sparse as sp

        n_cells = adata.n_obs
        gene_to_idx = {g: i for i, g in enumerate(adata.var_names)}

        X_new = np.zeros((n_cells, len(vocab)), dtype=np.float32)
        for col_idx, gene in enumerate(vocab):
            if gene in gene_to_idx:
                src = adata.X[:, gene_to_idx[gene]]
                if sp.issparse(src):
                    src = np.asarray(src.todense()).ravel()
                X_new[:, col_idx] = src

        import pandas as pd
        harmonized = AnnData(
            X=X_new,
            obs=adata.obs.copy(),
            var=pd.DataFrame(index=vocab),
        )
        return harmonized
