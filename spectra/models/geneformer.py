from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import scipy.sparse as sp
from anndata import AnnData

from spectra.models.base import InputType, ModelAdapter, ModelInfo

logger = logging.getLogger(__name__)

_HF_MODEL_ID = "ctheodoris/Geneformer"
_MAX_SEQ_LEN = 2048
_NORM_TARGET = 10_000
_EMBEDDING_DIM = 512  # 6L model


class GeneformerAdapter(ModelAdapter):
    """
    Geneformer 6L adapter (ctheodoris/Geneformer, 27.4M pretraining cells).

    Input: raw counts with Ensembl IDs in adata.var_names.
    CELLxGENE Census data already uses Ensembl IDs — no conversion needed.
    For data with gene symbols, convert var_names to Ensembl before calling.

    Tokenisation: normalise → rank by expression → map to token IDs → truncate to 2048.
    Embeddings: mean-pool last hidden layer over non-padding tokens → (n_cells, 512).

    Fine-tuning (Phase 2): trains a linear head on frozen transformer embeddings.
    Full transformer fine-tuning is deferred to Phase 3.
    """

    def __init__(
        self,
        model_path: str = _HF_MODEL_ID,
        batch_size: int = 32,
        device: Optional[str] = None,
        use_gene_median: bool = True,
        cache_dir: Optional[str] = None,
    ):
        self._model_path = model_path
        self._batch_size = batch_size
        self._cache_dir = cache_dir
        self._use_gene_median = use_gene_median
        self._device = device  # resolved lazily so import errors surface at load time
        self._model = None
        self._token_dict: Optional[dict] = None      # Ensembl ID → token ID
        self._gene_median_dict: Optional[dict] = None
        self._head = None  # logistic/linear head for fine_tune/predict

    # ------------------------------------------------------------------
    # ModelAdapter interface
    # ------------------------------------------------------------------

    @property
    def info(self) -> ModelInfo:
        return ModelInfo(
            name="geneformer",
            input_type=InputType.RANK,
            pretraining_cells=27_400_000,
            gene_vocabulary=None,  # ~25k Ensembl IDs — checked at runtime
            embedding_dim=_EMBEDDING_DIM,
            supports_fine_tuning=True,
            notes="6L, 512-dim. Ensembl IDs required in adata.var_names.",
        )

    def get_embeddings(self, adata: AnnData, layer: Optional[str] = None) -> np.ndarray:
        self._load()
        self._check_gene_overlap(adata)

        batches = []
        for start in range(0, adata.n_obs, self._batch_size):
            batch = adata[start : start + self._batch_size]
            batches.append(self._embed_batch(batch, layer))
        return np.vstack(batches).astype(np.float32)

    def fine_tune(
        self,
        adata_train: AnnData,
        task_type: str,
        label_key: str,
        n_epochs: int = 10,
        **kwargs,
    ) -> None:
        """Train a linear head on frozen Geneformer embeddings."""
        from sklearn.linear_model import LinearRegression, LogisticRegression

        embeddings = self.get_embeddings(adata_train)
        y = adata_train.obs[label_key].values

        if task_type.endswith("_regression"):
            self._head = LinearRegression()
        else:
            self._head = LogisticRegression(
                class_weight="balanced", max_iter=1000, random_state=42
            )
        self._head.fit(embeddings, y)
        logger.info("Geneformer: linear head trained on %d cells.", adata_train.n_obs)

    def predict(self, adata: AnnData, task_type: str) -> np.ndarray:
        if self._head is None:
            raise RuntimeError("Call fine_tune() before predict().")
        return self._head.predict(self.get_embeddings(adata))

    def reset(self) -> None:
        self._head = None

    # ------------------------------------------------------------------
    # Private: model loading
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if self._model is not None:
            return

        try:
            import torch
            from transformers import BertModel
            from huggingface_hub import hf_hub_download
        except ImportError as e:
            raise ImportError(
                "GeneformerAdapter requires torch and transformers. "
                "On EC2: install via setup_ec2.sh. "
                "Locally: conda install pytorch transformers -c pytorch -c conda-forge"
            ) from e

        self._device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Geneformer: loading model from %s on %s.", self._model_path, self._device)

        self._model = BertModel.from_pretrained(
            self._model_path,
            output_hidden_states=False,
            cache_dir=self._cache_dir,
        )
        self._model.eval()
        self._model.to(self._device)

        # Token dictionary: Ensembl ID → token ID
        token_path = hf_hub_download(
            self._model_path, "token_dictionary.pkl", cache_dir=self._cache_dir
        )
        with open(token_path, "rb") as f:
            self._token_dict = pickle.load(f)

        # Gene median dict: used to normalise by tissue-average expression
        if self._use_gene_median:
            try:
                median_path = hf_hub_download(
                    self._model_path, "gene_median_dictionary.pkl", cache_dir=self._cache_dir
                )
                with open(median_path, "rb") as f:
                    self._gene_median_dict = pickle.load(f)
            except Exception:
                logger.warning("Geneformer: gene_median_dictionary.pkl not found — skipping median normalisation.")
                self._gene_median_dict = None

    # ------------------------------------------------------------------
    # Private: tokenisation and embedding
    # ------------------------------------------------------------------

    def _check_gene_overlap(self, adata: AnnData, min_overlap: float = 0.05) -> None:
        vocab = set(self._token_dict.keys())
        present = set(adata.var_names) & vocab
        frac = len(present) / len(vocab)
        if frac < min_overlap:
            raise ValueError(
                f"Geneformer: gene overlap is {frac:.1%} ({len(present)}/{len(vocab)} "
                "vocabulary genes). Ensure adata.var_names are Ensembl IDs (ENSG…)."
            )
        if frac < 0.5:
            logger.warning("Geneformer: low gene overlap %.1f%% — embeddings may be unreliable.", frac * 100)

    def _tokenize_cell(self, counts: np.ndarray, var_names: np.ndarray) -> list[int]:
        """
        Convert a single cell's raw counts to a Geneformer token sequence.

        1. Normalise to NORM_TARGET sum.
        2. Optionally divide by gene median (captures tissue-level expression scale).
        3. Rank genes by normalised expression (highest first).
        4. Map to token IDs; skip genes not in vocabulary.
        5. Truncate to MAX_SEQ_LEN.
        """
        total = counts.sum()
        if total == 0:
            return []

        norm = counts * (_NORM_TARGET / total)

        if self._gene_median_dict is not None:
            medians = np.array([
                self._gene_median_dict.get(g, 1.0) for g in var_names
            ], dtype=np.float32)
            medians = np.where(medians > 0, medians, 1.0)
            norm = norm / medians

        # Only rank genes with non-zero expression
        nonzero_mask = norm > 0
        ranked_idx = np.argsort(-norm[nonzero_mask])
        ranked_genes = var_names[nonzero_mask][ranked_idx]

        tokens = [
            self._token_dict[g] for g in ranked_genes if g in self._token_dict
        ]
        return tokens[:_MAX_SEQ_LEN]

    def _embed_batch(self, adata: AnnData, layer: Optional[str]) -> np.ndarray:
        import torch

        X = adata.layers[layer] if layer is not None else adata.X
        if sp.issparse(X):
            X = X.toarray()
        X = X.astype(np.float32)

        var_names = np.array(adata.var_names)
        token_seqs = [self._tokenize_cell(X[i], var_names) for i in range(X.shape[0])]

        # Pad sequences
        lengths = [len(s) for s in token_seqs]
        max_len = max(lengths) if lengths else 1
        n = len(token_seqs)

        input_ids = torch.zeros(n, max_len, dtype=torch.long, device=self._device)
        attention_mask = torch.zeros(n, max_len, dtype=torch.long, device=self._device)
        for i, (seq, length) in enumerate(zip(token_seqs, lengths)):
            if length:
                input_ids[i, :length] = torch.tensor(seq, dtype=torch.long)
                attention_mask[i, :length] = 1

        with torch.no_grad():
            outputs = self._model(input_ids=input_ids, attention_mask=attention_mask)
            hidden = outputs.last_hidden_state  # (n, seq_len, 512)
            mask = attention_mask.unsqueeze(-1).float()
            # Mean-pool over non-padding positions
            embeddings = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        # .tolist() sidesteps NumPy 1.x/2.x ABI mismatch with older torch builds.
        # On EC2 with a matching torch + NumPy build, .numpy() would also work.
        return np.array(embeddings.cpu().tolist(), dtype=np.float32)
