from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Cell type names as they appear in CELLxGENE Census (Cell Ontology labels).
# These are the liver types we care about — others are filtered out.
_LIVER_CELL_TYPES = {
    "hepatocyte",
    "Kupffer cell",
    "liver sinusoidal endothelial cell",
    "hepatic stellate cell",
    "cholangiocyte",
    "portal fibroblast",
    "intrahepatic cholangiocyte",
    "periportal region hepatocyte",
}

# Maps CXG Cell Ontology labels → Spectra canonical names used in our tasks.
CXG_TO_SPECTRA: dict[str, str] = {
    "hepatocyte": "Hepatocytes",
    "periportal region hepatocyte": "Hepatocytes",
    "Kupffer cell": "Kupffer cells",
    "liver sinusoidal endothelial cell": "LSECs",
    "hepatic stellate cell": "Hepatic stellate cells",
    "cholangiocyte": "Cholangiocytes",
    "intrahepatic cholangiocyte": "Cholangiocytes",
    "portal fibroblast": "Portal fibroblasts",
}


def load_cellxgene_liver(
    max_cells: int = 8_000,
    seed: int = 42,
    cache_path: Optional[str | Path] = None,
) -> "anndata.AnnData":
    """
    Download a liver slice from CELLxGENE Census and return a ready-to-use AnnData.

    Parameters
    ----------
    max_cells
        Maximum number of cells to return (random sample). Defaults to 8 000.
    seed
        Random seed for the cell sample.
    cache_path
        If provided, save/load the AnnData as a .h5ad file at this path to avoid
        re-downloading. The path is deliberately kept out of the git repo (data/ is
        gitignored).

    Returns
    -------
    AnnData with:
        - X: raw counts (int32)
        - var_names: Ensembl gene IDs (ENSG…) — ready for Geneformer
        - obs["cell_type"]: Spectra canonical cell type names
        - obs["cell_type_cxg"]: original CELLxGENE Cell Ontology label
        - obs["dataset_id"]: source dataset
    """
    try:
        import cellxgene_census
    except ImportError:
        raise ImportError(
            "cellxgene_census is required: pip install cellxgene-census"
        ) from None

    import anndata

    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.exists():
            logger.info("Loading cached liver dataset from %s", cache_path)
            return anndata.read_h5ad(cache_path)

    logger.info("Opening CELLxGENE Census (this may take a moment)…")
    with cellxgene_census.open_soma() as census:
        adata = cellxgene_census.get_anndata(
            census,
            organism="Homo sapiens",
            obs_value_filter=(
                "tissue_general == 'liver'"
                " and is_primary_data == True"
                " and cell_type in ["
                + ", ".join(f"'{ct}'" for ct in _LIVER_CELL_TYPES)
                + "]"
            ),
            var_value_filter="feature_biotype == 'gene'",
            obs_column_names=["cell_type", "dataset_id", "tissue", "sex", "disease"],
        )

    logger.info("Census returned %d cells before sampling.", adata.n_obs)

    # Random subsample
    if adata.n_obs > max_cells:
        rng = np.random.default_rng(seed)
        idx = rng.choice(adata.n_obs, size=max_cells, replace=False)
        idx.sort()
        adata = adata[idx].copy()

    # Add canonical cell type names alongside the raw CXG labels
    adata.obs["cell_type_cxg"] = adata.obs["cell_type"].astype(str)
    adata.obs["cell_type"] = (
        adata.obs["cell_type_cxg"].map(CXG_TO_SPECTRA).fillna("Other")
    )

    # Drop "Other" (unmapped types)
    keep = adata.obs["cell_type"] != "Other"
    if (~keep).sum() > 0:
        logger.warning(
            "Dropping %d cells with unmapped cell types: %s",
            (~keep).sum(),
            adata.obs.loc[~keep, "cell_type_cxg"].unique().tolist(),
        )
    adata = adata[keep].copy()

    logger.info(
        "Final dataset: %d cells, %d genes. Cell type counts:\n%s",
        adata.n_obs,
        adata.n_vars,
        adata.obs["cell_type"].value_counts().to_string(),
    )

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        adata.write_h5ad(cache_path)
        logger.info("Cached to %s", cache_path)

    return adata
