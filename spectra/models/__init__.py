from spectra.models.base import InputType, ModelAdapter, ModelInfo
from spectra.models.mock import MockModelAdapter
from spectra.models.pca_baseline import PCABaselineAdapter

# GeneformerAdapter imported lazily — requires torch + transformers (Phase 2+)
def __getattr__(name: str):
    if name == "GeneformerAdapter":
        from spectra.models.geneformer import GeneformerAdapter
        return GeneformerAdapter
    raise AttributeError(f"module 'spectra.models' has no attribute {name!r}")

__all__ = ["InputType", "ModelAdapter", "ModelInfo", "MockModelAdapter", "PCABaselineAdapter", "GeneformerAdapter"]
