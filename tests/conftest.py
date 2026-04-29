import numpy as np
import pytest
import scipy.sparse as sp
import anndata

from spectra.models import MockModelAdapter, PCABaselineAdapter


@pytest.fixture
def liver_adata():
    """200-cell synthetic liver dataset with realistic cell type proportions."""
    np.random.seed(0)
    n = 200
    X = sp.random(n, 500, density=0.1, format="csr", random_state=0)
    adata = anndata.AnnData(X=X)
    # hepatocytes dominate — tests macro-F1 vs accuracy distinction
    adata.obs["cell_type"] = (
        ["Hepatocytes"] * 80
        + ["Kupffer cells"] * 20
        + ["LSECs"] * 40
        + ["Cholangiocytes"] * 30
        + ["Hepatic stellate cells"] * 20
        + ["Portal fibroblasts"] * 10
    )
    return adata


@pytest.fixture
def zone_adata():
    """200-cell synthetic dataset with zone labels and a continuous zone score."""
    np.random.seed(1)
    n = 200
    X = sp.random(n, 500, density=0.1, format="csr", random_state=1)
    adata = anndata.AnnData(X=X)
    adata.obs["zone"] = ["periportal"] * 70 + ["mid"] * 60 + ["pericentral"] * 70
    # Synthetic continuous score derived from zone + small noise
    base = {"periportal": 0.0, "mid": 0.5, "pericentral": 1.0}
    scores = np.array([base[z] for z in adata.obs["zone"]], dtype=np.float32)
    scores += np.random.default_rng(1).normal(0, 0.05, size=n).astype(np.float32)
    scores = np.clip(scores, 0.0, 1.0)
    adata.obs["zone_score"] = scores
    return adata


@pytest.fixture
def mock_adapter():
    return MockModelAdapter(embedding_dim=32, seed=42)


@pytest.fixture
def pca_adapter():
    return PCABaselineAdapter(n_components=10, seed=42)
