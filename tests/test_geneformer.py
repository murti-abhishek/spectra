"""
Tests for GeneformerAdapter.

All tests mock out model loading so they run on CPU without downloading weights.
GPU / full integration tests require Phase 2 EC2 environment.
"""
from __future__ import annotations

import pickle
import numpy as np
import pandas as pd
import pytest
import anndata
import scipy.sparse as sp
from unittest.mock import MagicMock, patch


# ── Fixtures ──────────────────────────────────────────────────────────────────

N_VOCAB = 100  # small fake vocabulary for tests

@pytest.fixture
def fake_vocab():
    """Fake Ensembl ID token dictionary."""
    genes = [f"ENSG{i:011d}" for i in range(N_VOCAB)]
    return {g: i + 5 for i, g in enumerate(genes)}  # token IDs start at 5


@pytest.fixture
def ensg_adata(fake_vocab):
    """AnnData with Ensembl IDs in var_names — matches Geneformer vocabulary."""
    np.random.seed(42)
    genes = list(fake_vocab.keys())
    X = sp.random(30, N_VOCAB, density=0.2, format="csr", random_state=42)
    adata = anndata.AnnData(X=X, var=pd.DataFrame(index=genes))
    adata.obs["cell_type"] = ["Hepatocytes"] * 15 + ["Kupffer cells"] * 15
    adata.obs["zone"] = ["periportal"] * 10 + ["mid"] * 10 + ["pericentral"] * 10
    return adata


@pytest.fixture
def loaded_adapter(fake_vocab):
    """GeneformerAdapter with model loading mocked out."""
    from spectra.models.geneformer import GeneformerAdapter

    adapter = GeneformerAdapter(batch_size=16, device="cpu")

    # Inject fake token dict and a mock BERT model
    adapter._token_dict = fake_vocab
    adapter._gene_median_dict = None

    mock_model = MagicMock()
    # Simulate BertModel output: last_hidden_state (batch, seq_len, 512)
    def fake_forward(input_ids, attention_mask):
        b, s = input_ids.shape
        out = MagicMock()
        import torch
        out.last_hidden_state = torch.randn(b, s, 512)
        return out

    mock_model.side_effect = fake_forward
    mock_model.__call__ = fake_forward
    adapter._model = mock_model
    adapter._device = "cpu"
    return adapter


# ── ModelInfo ─────────────────────────────────────────────────────────────────

def test_info_fields():
    from spectra.models.geneformer import GeneformerAdapter, _EMBEDDING_DIM
    from spectra.models.base import InputType
    adapter = GeneformerAdapter()
    info = adapter.info
    assert info.name == "geneformer"
    assert info.input_type == InputType.RANK
    assert info.pretraining_cells == 27_400_000
    assert info.embedding_dim == _EMBEDDING_DIM
    assert info.supports_fine_tuning is True


# ── Tokenisation ──────────────────────────────────────────────────────────────

def test_tokenize_cell_returns_token_ids(loaded_adapter, ensg_adata):
    counts = np.asarray(ensg_adata.X[0].todense()).ravel()
    var_names = np.array(ensg_adata.var_names)
    tokens = loaded_adapter._tokenize_cell(counts, var_names)
    assert isinstance(tokens, list)
    assert all(isinstance(t, int) for t in tokens)

def test_tokenize_empty_cell_returns_empty(loaded_adapter, ensg_adata):
    counts = np.zeros(N_VOCAB, dtype=np.float32)
    tokens = loaded_adapter._tokenize_cell(counts, np.array(ensg_adata.var_names))
    assert tokens == []

def test_tokenize_truncates_to_max_seq_len(loaded_adapter, fake_vocab):
    from spectra.models.geneformer import _MAX_SEQ_LEN, GeneformerAdapter
    adapter = GeneformerAdapter.__new__(GeneformerAdapter)
    adapter._token_dict = fake_vocab
    adapter._gene_median_dict = None
    counts = np.ones(N_VOCAB, dtype=np.float32)
    var_names = np.array(list(fake_vocab.keys()))
    tokens = adapter._tokenize_cell(counts, var_names)
    assert len(tokens) <= min(_MAX_SEQ_LEN, N_VOCAB)

def test_tokenize_genes_ranked_by_expression(loaded_adapter, fake_vocab):
    """Higher-expression genes should appear earlier in the token sequence."""
    var_names = np.array(list(fake_vocab.keys()))
    counts = np.zeros(N_VOCAB, dtype=np.float32)
    # Gene at index 5 has highest expression, gene at index 2 second
    counts[5] = 100.0
    counts[2] = 50.0
    tokens = loaded_adapter._tokenize_cell(counts, var_names)
    assert tokens[0] == fake_vocab[var_names[5]]
    assert tokens[1] == fake_vocab[var_names[2]]


# ── Gene overlap check ────────────────────────────────────────────────────────

def test_check_gene_overlap_raises_below_min(loaded_adapter):
    bad_adata = anndata.AnnData(
        X=np.ones((5, 3)),
        var=pd.DataFrame(index=["FAKE1", "FAKE2", "FAKE3"]),
    )
    with pytest.raises(ValueError, match="gene overlap"):
        loaded_adapter._check_gene_overlap(bad_adata)

def test_check_gene_overlap_passes_with_full_vocab(loaded_adapter, ensg_adata):
    loaded_adapter._check_gene_overlap(ensg_adata)  # should not raise


# ── Embeddings ────────────────────────────────────────────────────────────────

def test_get_embeddings_shape(loaded_adapter, ensg_adata):
    import torch
    # Patch the model call to return proper tensors
    def fake_call(input_ids, attention_mask):
        b, s = input_ids.shape
        out = MagicMock()
        out.last_hidden_state = torch.randn(b, s, 512)
        return out
    loaded_adapter._model = fake_call
    emb = loaded_adapter.get_embeddings(ensg_adata)
    assert emb.shape == (30, 512)
    assert emb.dtype == np.float32

def test_get_embeddings_batches_correctly(loaded_adapter, ensg_adata):
    import torch
    calls = []
    def counting_call(input_ids, attention_mask):
        calls.append(input_ids.shape[0])
        b, s = input_ids.shape
        out = MagicMock()
        out.last_hidden_state = torch.randn(b, s, 512)
        return out
    loaded_adapter._model = counting_call
    loaded_adapter._batch_size = 10
    loaded_adapter.get_embeddings(ensg_adata)
    assert len(calls) == 3  # 30 cells / batch_size 10 = 3 batches
    assert sum(calls) == 30


# ── Fine-tune and predict ─────────────────────────────────────────────────────

def test_fine_tune_and_predict(loaded_adapter, ensg_adata):
    import torch
    def fake_call(input_ids, attention_mask):
        b, s = input_ids.shape
        out = MagicMock()
        out.last_hidden_state = torch.randn(b, s, 512)
        return out
    loaded_adapter._model = fake_call
    train, test = ensg_adata[:20], ensg_adata[20:]
    loaded_adapter.fine_tune(train, "liver_cell_type", "cell_type")
    preds = loaded_adapter.predict(test, "liver_cell_type")
    assert len(preds) == 10
    known = set(ensg_adata.obs["cell_type"].unique())
    assert all(p in known for p in preds)

def test_predict_before_fine_tune_raises():
    from spectra.models.geneformer import GeneformerAdapter
    adapter = GeneformerAdapter.__new__(GeneformerAdapter)
    adapter._head = None
    with pytest.raises(RuntimeError, match="fine_tune"):
        adapter.predict(MagicMock(), "liver_cell_type")

def test_reset_clears_head(loaded_adapter, ensg_adata):
    import torch
    def fake_call(input_ids, attention_mask):
        b, s = input_ids.shape
        out = MagicMock()
        out.last_hidden_state = torch.randn(b, s, 512)
        return out
    loaded_adapter._model = fake_call
    loaded_adapter.fine_tune(ensg_adata, "liver_cell_type", "cell_type")
    assert loaded_adapter._head is not None
    loaded_adapter.reset()
    assert loaded_adapter._head is None


# ── Load guard ────────────────────────────────────────────────────────────────

def test_load_raises_without_torch(monkeypatch):
    from spectra.models.geneformer import GeneformerAdapter
    adapter = GeneformerAdapter.__new__(GeneformerAdapter)
    adapter._model = None
    adapter._token_dict = None
    adapter._gene_median_dict = None
    adapter._cache_dir = None
    adapter._use_gene_median = False
    adapter._model_path = "ctheodoris/Geneformer"
    adapter._device = None

    import builtins
    real_import = builtins.__import__
    def blocked_import(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("torch not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    with pytest.raises(ImportError, match="torch"):
        adapter._load()
