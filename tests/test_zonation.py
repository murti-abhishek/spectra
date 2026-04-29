import numpy as np
import pytest

from spectra.tasks.zonation_prediction import ZonationPredictionTask
from spectra.tasks.base import EvalMode


class TestZonationClassification:
    def test_get_splits_count(self, zone_adata):
        task = ZonationPredictionTask(zone_adata, "zone", mode="classification")
        splits = task.get_splits(n_folds=5, seed=0)
        assert len(splits) == 5

    def test_splits_stratified(self, zone_adata):
        task = ZonationPredictionTask(zone_adata, "zone", mode="classification")
        splits = task.get_splits(n_folds=3, seed=0)
        for s in splits:
            present = set(s.adata_test.obs["zone"].unique())
            assert present == {"periportal", "mid", "pericentral"}

    def test_evaluate_returns_required_keys(self, zone_adata):
        task = ZonationPredictionTask(zone_adata, "zone", mode="classification")
        y = zone_adata.obs["zone"].values
        metrics = task.evaluate(y, y)
        assert "macro_f1" in metrics
        assert "accuracy" in metrics
        assert "cohen_kappa" in metrics
        assert "f1_periportal" in metrics
        assert "f1_mid" in metrics
        assert "f1_pericentral" in metrics

    def test_perfect_classification_gives_f1_one(self, zone_adata):
        task = ZonationPredictionTask(zone_adata, "zone", mode="classification")
        y = zone_adata.obs["zone"].values
        metrics = task.evaluate(y, y)
        assert metrics["macro_f1"] == pytest.approx(1.0)

    def test_get_probe_targets_returns_strings(self, zone_adata):
        task = ZonationPredictionTask(zone_adata, "zone", mode="classification")
        targets = task.get_probe_targets(zone_adata, "zone")
        assert targets.dtype.kind in ("U", "O")  # unicode or object (string)


class TestZonationRegression:
    def test_get_splits_count(self, zone_adata):
        task = ZonationPredictionTask(zone_adata, "zone", mode="regression")
        splits = task.get_splits(n_folds=5, seed=0)
        assert len(splits) == 5

    def test_evaluate_returns_required_keys(self, zone_adata):
        task = ZonationPredictionTask(zone_adata, "zone", mode="regression", score_key="zone_score")
        y = zone_adata.obs["zone_score"].values.astype(np.float32)
        metrics = task.evaluate(y, y)
        assert "spearman_r" in metrics
        assert "pearson_r" in metrics
        assert "r2" in metrics
        assert "rmse" in metrics

    def test_perfect_regression_gives_spearman_one(self, zone_adata):
        task = ZonationPredictionTask(zone_adata, "zone", mode="regression", score_key="zone_score")
        y = zone_adata.obs["zone_score"].values.astype(np.float32)
        metrics = task.evaluate(y, y)
        assert metrics["spearman_r"] == pytest.approx(1.0)
        assert metrics["rmse"] == pytest.approx(0.0, abs=1e-5)

    def test_get_probe_targets_with_score_key(self, zone_adata):
        task = ZonationPredictionTask(zone_adata, "zone", mode="regression", score_key="zone_score")
        targets = task.get_probe_targets(zone_adata, "zone")
        assert targets.dtype == np.float32
        assert np.all((targets >= 0.0) & (targets <= 1.0))

    def test_derive_zone_score_from_labels(self, zone_adata):
        task = ZonationPredictionTask(zone_adata, "zone", mode="regression")
        targets = task.get_probe_targets(zone_adata, "zone")
        assert targets.dtype == np.float32
        # periportal → 0, mid → 0.5, pericentral → 1
        pp_mask = zone_adata.obs["zone"].values == "periportal"
        assert np.all(targets[pp_mask] == pytest.approx(0.0))

    def test_derive_zone_score_static(self):
        labels = np.array(["periportal", "mid", "pericentral", "periportal"])
        scores = ZonationPredictionTask.derive_zone_score(labels)
        assert scores.tolist() == pytest.approx([0.0, 0.5, 1.0, 0.0])

    def test_missing_score_key_raises(self, zone_adata):
        with pytest.raises(KeyError, match="score_key"):
            ZonationPredictionTask(zone_adata, "zone", mode="regression", score_key="nonexistent")


class TestZonationValidation:
    def test_invalid_mode_raises(self, zone_adata):
        with pytest.raises(ValueError, match="mode"):
            ZonationPredictionTask(zone_adata, "zone", mode="unsupported")

    def test_missing_label_key_raises(self, zone_adata):
        with pytest.raises(KeyError):
            ZonationPredictionTask(zone_adata, "nonexistent", mode="classification")

    def test_tissue_specific_flag(self, zone_adata):
        task = ZonationPredictionTask(zone_adata, "zone")
        assert task.tissue_specific is True
        assert task.target_tissue == "liver"
