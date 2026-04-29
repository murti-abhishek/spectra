import numpy as np
import pandas as pd
import pytest
import anndata
import scipy.sparse as sp

from spectra.models.base import InputType, ModelAdapter, ModelInfo
from spectra.models import PCABaselineAdapter


# Minimal concrete adapter to exercise _harmonize_genes
class _VocabAdapter(ModelAdapter):
    def __init__(self, vocab):
        self._vocab = vocab

    @property
    def info(self):
        return ModelInfo(
            name="vocab_test",
            input_type=InputType.RAW,
            gene_vocabulary=self._vocab,
        )

    def get_embeddings(self, adata, layer=None):
        harmonized = self._harmonize_genes(adata)
        return np.zeros((harmonized.n_obs, len(self._vocab)), dtype=np.float32)

    def fine_tune(self, adata_train, task_type, label_key, n_epochs=10, **kwargs):
        pass

    def predict(self, adata, task_type):
        return np.zeros(adata.n_obs, dtype=np.float32)


def _make_adata(genes, n=20):
    X = sp.eye(n, len(genes), format="csr")
    return anndata.AnnData(X=X, var=pd.DataFrame(index=genes))


class TestHarmonizeGenes:
    def test_full_overlap_returns_correct_shape(self):
        vocab = [f"gene_{i}" for i in range(50)]
        adapter = _VocabAdapter(vocab)
        adata = _make_adata(vocab)
        out = adapter._harmonize_genes(adata)
        assert out.shape == (adata.n_obs, 50)

    def test_partial_overlap_fills_missing_with_zero(self):
        vocab = [f"gene_{i}" for i in range(50)]
        adapter = _VocabAdapter(vocab)
        # adata only has first 30 genes
        adata = _make_adata(vocab[:30])
        out = adapter._harmonize_genes(adata)
        assert out.shape == (adata.n_obs, 50)
        # last 20 columns should be zero-filled
        assert np.all(out.X[:, 30:] == 0.0)

    def test_below_min_overlap_raises(self):
        vocab = [f"gene_{i}" for i in range(1000)]
        adapter = _VocabAdapter(vocab)
        # adata has only 10 of 1000 vocab genes → 1% overlap → below 5% floor
        adata = _make_adata(vocab[:10])
        with pytest.raises(ValueError, match="gene overlap"):
            adapter._harmonize_genes(adata)

    def test_none_vocabulary_returns_adata_unchanged(self):
        class _UniversalAdapter(_VocabAdapter):
            @property
            def info(self):
                return ModelInfo(name="universal", input_type=InputType.UNIVERSAL, gene_vocabulary=None)

        adapter = _UniversalAdapter(vocab=None)
        adata = _make_adata([f"g{i}" for i in range(20)])
        out = adapter._harmonize_genes(adata)
        assert out is adata  # same object returned

    def test_sparse_input_handled(self):
        vocab = [f"gene_{i}" for i in range(50)]
        adapter = _VocabAdapter(vocab)
        X = sp.random(20, 50, density=0.2, format="csr", random_state=0)
        adata = anndata.AnnData(X=X, var=pd.DataFrame(index=vocab))
        out = adapter._harmonize_genes(adata)
        assert out.X.shape == (20, 50)


class TestMockAdapter:
    def test_embeddings_shape_and_dtype(self, mock_adapter, liver_adata):
        emb = mock_adapter.get_embeddings(liver_adata)
        assert emb.shape == (liver_adata.n_obs, 32)
        assert emb.dtype == np.float32

    def test_fine_tune_stores_classes(self, mock_adapter, liver_adata):
        mock_adapter.fine_tune(liver_adata, "liver_cell_type", "cell_type")
        assert mock_adapter._classes is not None
        assert set(mock_adapter._classes) == set(liver_adata.obs["cell_type"].unique())

    def test_predict_classification_returns_known_classes(self, mock_adapter, liver_adata):
        mock_adapter.fine_tune(liver_adata, "liver_cell_type", "cell_type")
        preds = mock_adapter.predict(liver_adata, "liver_cell_type")
        known = set(liver_adata.obs["cell_type"].unique())
        assert all(p in known for p in preds)

    def test_predict_regression_returns_floats_in_unit_interval(self, mock_adapter, liver_adata):
        preds = mock_adapter.predict(liver_adata, "zonation_regression")
        assert preds.dtype == np.float32
        assert preds.min() >= 0.0 and preds.max() <= 1.0

    def test_reset_clears_classes(self, mock_adapter, liver_adata):
        mock_adapter.fine_tune(liver_adata, "liver_cell_type", "cell_type")
        mock_adapter.reset()
        assert mock_adapter._classes is None


class TestPCAAdapter:
    def test_first_call_fits_subsequent_transforms(self, pca_adapter, liver_adata):
        train = liver_adata[:150]
        test = liver_adata[150:]
        emb_train = pca_adapter.get_embeddings(train)
        emb_test = pca_adapter.get_embeddings(test)
        assert emb_train.shape == (150, 10)
        assert emb_test.shape == (50, 10)
        assert pca_adapter._pca is not None

    def test_reset_clears_pca_and_head(self, pca_adapter, liver_adata):
        pca_adapter.get_embeddings(liver_adata)
        pca_adapter.reset()
        assert pca_adapter._pca is None
        assert pca_adapter._head is None

    def test_fine_tune_then_predict(self, pca_adapter, liver_adata):
        train, test = liver_adata[:150], liver_adata[150:]
        pca_adapter.fine_tune(train, "liver_cell_type", "cell_type")
        preds = pca_adapter.predict(test, "liver_cell_type")
        assert len(preds) == 50
        known = set(liver_adata.obs["cell_type"].unique())
        assert all(p in known for p in preds)

    def test_predict_before_fine_tune_raises(self, pca_adapter, liver_adata):
        with pytest.raises(RuntimeError, match="fine_tune"):
            pca_adapter.predict(liver_adata, "liver_cell_type")

    def test_n_components_capped_at_n_cells(self):
        # 5 cells, requesting 10 components → capped to 4
        adapter = PCABaselineAdapter(n_components=10)
        X = sp.eye(5, 50, format="csr")
        adata = anndata.AnnData(X=X)
        emb = adapter.get_embeddings(adata)
        assert emb.shape[0] == 5
        assert emb.shape[1] <= 4
