import numpy as np
import pytest
import anndata
import scipy.sparse as sp

from spectra.tasks.liver_cell_type_annotation import LiverCellTypeAnnotationTask
from spectra.tasks.base import EvalMode


class TestLiverCellTypeAnnotationTask:
    def test_get_splits_count(self, liver_adata):
        task = LiverCellTypeAnnotationTask(liver_adata, "cell_type")
        splits = task.get_splits(n_folds=5, seed=0)
        assert len(splits) == 5

    def test_splits_are_disjoint_and_complete(self, liver_adata):
        task = LiverCellTypeAnnotationTask(liver_adata, "cell_type")
        splits = task.get_splits(n_folds=5, seed=0)
        all_test_indices = []
        for s in splits:
            all_test_indices.extend(list(s.adata_test.obs.index))
        assert len(all_test_indices) == liver_adata.n_obs
        assert len(set(all_test_indices)) == liver_adata.n_obs

    def test_split_label_key_propagated(self, liver_adata):
        task = LiverCellTypeAnnotationTask(liver_adata, "cell_type")
        splits = task.get_splits(n_folds=3)
        assert all(s.label_key == "cell_type" for s in splits)

    def test_fold_indices_assigned(self, liver_adata):
        task = LiverCellTypeAnnotationTask(liver_adata, "cell_type")
        splits = task.get_splits(n_folds=4)
        assert [s.fold for s in splits] == [0, 1, 2, 3]

    def test_evaluate_returns_required_keys(self, liver_adata):
        task = LiverCellTypeAnnotationTask(liver_adata, "cell_type")
        y_true = liver_adata.obs["cell_type"].values
        metrics = task.evaluate(y_true, y_true)  # perfect predictions
        assert "macro_f1" in metrics
        assert "accuracy" in metrics
        assert "cohen_kappa" in metrics

    def test_perfect_predictions_give_macro_f1_one(self, liver_adata):
        task = LiverCellTypeAnnotationTask(liver_adata, "cell_type")
        y = liver_adata.obs["cell_type"].values
        metrics = task.evaluate(y, y)
        assert metrics["macro_f1"] == pytest.approx(1.0)
        assert metrics["accuracy"] == pytest.approx(1.0)

    def test_per_class_keys_are_sanitised(self, liver_adata):
        task = LiverCellTypeAnnotationTask(liver_adata, "cell_type")
        y = liver_adata.obs["cell_type"].values
        metrics = task.evaluate(y, y)
        assert "f1_kupffer_cells" in metrics
        assert "f1_hepatic_stellate_cells" in metrics
        assert "f1_lsecs" in metrics
        # no spaces or uppercase in keys
        for key in metrics:
            assert " " not in key
            assert key == key.lower()

    def test_invalid_resolution_raises(self, liver_adata):
        with pytest.raises(ValueError, match="resolution"):
            LiverCellTypeAnnotationTask(liver_adata, "cell_type", resolution="zone")

    def test_missing_label_key_raises(self, liver_adata):
        with pytest.raises(KeyError):
            LiverCellTypeAnnotationTask(liver_adata, "nonexistent_column")

    def test_unexpected_labels_warn(self, liver_adata):
        adata = liver_adata.copy()
        labels = adata.obs["cell_type"].tolist()
        labels[0] = "Mystery cell"
        adata.obs["cell_type"] = labels
        # Should not raise — unexpected labels produce a warning log, not an exception
        task = LiverCellTypeAnnotationTask(adata, "cell_type")
        assert task is not None

    def test_supported_modes(self, liver_adata):
        task = LiverCellTypeAnnotationTask(liver_adata, "cell_type")
        assert EvalMode.ZERO_SHOT in task.supported_modes
        assert EvalMode.FINE_TUNED in task.supported_modes

    def test_tissue_specific_flag(self, liver_adata):
        task = LiverCellTypeAnnotationTask(liver_adata, "cell_type")
        assert task.tissue_specific is True
        assert task.target_tissue == "liver"
