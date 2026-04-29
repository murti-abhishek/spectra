import json
import numpy as np
import pytest
from pathlib import Path

from spectra.models import MockModelAdapter, PCABaselineAdapter
from spectra.tasks import LiverCellTypeAnnotationTask, ZonationPredictionTask
from spectra.tasks.base import EvalMode, TaskResult
from spectra.runner import BenchmarkRunner, RunConfig


@pytest.fixture
def run_config(tmp_path):
    return RunConfig(output_dir=tmp_path, n_splits=3, zero_shot=True, fine_tuned=False)


@pytest.fixture
def tasks(liver_adata, zone_adata):
    return [
        LiverCellTypeAnnotationTask(liver_adata, "cell_type"),
        ZonationPredictionTask(zone_adata, "zone", mode="classification"),
    ]


class TestRunConfig:
    def test_modes_zero_shot_only(self, tmp_path):
        cfg = RunConfig(output_dir=tmp_path, zero_shot=True, fine_tuned=False)
        assert cfg.modes == [EvalMode.ZERO_SHOT]

    def test_modes_fine_tuned_only(self, tmp_path):
        cfg = RunConfig(output_dir=tmp_path, zero_shot=False, fine_tuned=True)
        assert cfg.modes == [EvalMode.FINE_TUNED]

    def test_modes_both(self, tmp_path):
        cfg = RunConfig(output_dir=tmp_path, zero_shot=True, fine_tuned=True)
        assert EvalMode.ZERO_SHOT in cfg.modes
        assert EvalMode.FINE_TUNED in cfg.modes

    def test_output_dir_created(self, tmp_path):
        out = tmp_path / "nested" / "dir"
        runner = BenchmarkRunner([], [], RunConfig(output_dir=out))
        assert out.exists()


class TestBenchmarkRunner:
    def test_result_count(self, liver_adata, run_config, tasks):
        models = [MockModelAdapter()]
        runner = BenchmarkRunner(models, tasks, run_config)
        results = runner.run()
        # 1 model × 2 tasks × 3 folds × 1 mode = 6
        assert len(results) == 6

    def test_results_have_correct_structure(self, liver_adata, run_config, tasks):
        runner = BenchmarkRunner([MockModelAdapter()], tasks, run_config)
        results = runner.run()
        for r in results:
            assert isinstance(r, TaskResult)
            assert r.model == "mock"
            assert r.mode == EvalMode.ZERO_SHOT
            assert "macro_f1" in r.metrics

    def test_checkpointing_skips_existing(self, liver_adata, run_config, tasks):
        models = [MockModelAdapter()]
        runner1 = BenchmarkRunner(models, tasks, run_config)
        results1 = runner1.run()

        # Verify JSON files were written
        json_files = list(run_config.output_dir.rglob("*.json"))
        assert len(json_files) == 6

        # Second run must load from cache (results identical)
        runner2 = BenchmarkRunner(models, tasks, run_config)
        results2 = runner2.run()
        assert len(results2) == len(results1)
        for r1, r2 in zip(results1, results2):
            assert r1.metrics == r2.metrics

    def test_result_json_contents(self, liver_adata, run_config, tasks):
        runner = BenchmarkRunner([MockModelAdapter()], [tasks[0]], run_config)
        runner.run()
        json_files = list(run_config.output_dir.rglob("*.json"))
        assert json_files
        data = json.loads(json_files[0].read_text())
        assert {"model", "task", "fold", "mode", "metrics"} <= data.keys()
        assert data["model"] == "mock"

    def test_multiple_models(self, liver_adata, run_config, tasks):
        models = [MockModelAdapter(), PCABaselineAdapter(n_components=10)]
        runner = BenchmarkRunner(models, [tasks[0]], run_config)
        results = runner.run()
        model_names = {r.model for r in results}
        assert model_names == {"mock", "pca_baseline"}

    def test_fine_tuned_mode(self, liver_adata, zone_adata, tmp_path):
        cfg = RunConfig(output_dir=tmp_path, n_splits=3, zero_shot=False, fine_tuned=True)
        task = LiverCellTypeAnnotationTask(liver_adata, "cell_type")
        runner = BenchmarkRunner([PCABaselineAdapter(n_components=10)], [task], cfg)
        results = runner.run()
        assert len(results) == 3
        assert all(r.mode == EvalMode.FINE_TUNED for r in results)

    def test_skips_fine_tuned_for_unsupported_model(self, liver_adata, tmp_path):
        from spectra.models.base import ModelInfo, InputType
        from spectra.models.mock import MockModelAdapter as _Base

        class NoFTAdapter(_Base):
            @property
            def info(self):
                return ModelInfo(name="no_ft", input_type=InputType.RAW,
                                 supports_fine_tuning=False)

        cfg = RunConfig(output_dir=tmp_path, n_splits=3, zero_shot=False, fine_tuned=True)
        task = LiverCellTypeAnnotationTask(liver_adata, "cell_type")
        runner = BenchmarkRunner([NoFTAdapter()], [task], cfg)
        results = runner.run()
        # Results returned but metrics are empty dicts
        assert all(r.metrics == {} for r in results)
