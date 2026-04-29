from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from spectra.models.base import ModelAdapter
from spectra.tasks.base import EvalMode, Task, TaskResult

logger = logging.getLogger(__name__)


@dataclass
class RunConfig:
    output_dir: Path = Path("results")
    n_splits: int = 5
    seed: int = 42
    zero_shot: bool = True
    fine_tuned: bool = True
    modes: list[EvalMode] = field(init=False)

    def __post_init__(self):
        self.output_dir = Path(self.output_dir)
        self.modes = []
        if self.zero_shot:
            self.modes.append(EvalMode.ZERO_SHOT)
        if self.fine_tuned:
            self.modes.append(EvalMode.FINE_TUNED)


class BenchmarkRunner:
    """
    Orchestrates model × task × fold evaluation.

    Checkpointing: each (model, task, fold, mode) result is written to
    results/{model}/{task}_fold{fold}_{mode}.json immediately after completion.
    Re-running skips any result that already exists on disk.
    """

    def __init__(
        self,
        models: list[ModelAdapter],
        tasks: list[Task],
        config: Optional[RunConfig] = None,
    ):
        self.models = models
        self.tasks = tasks
        self.config = config or RunConfig()
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> list[TaskResult]:
        all_results: list[TaskResult] = []

        total = len(self.models) * len(self.tasks) * self.config.n_splits * len(self.config.modes)
        pbar = tqdm(total=total, desc="Benchmark", unit="eval")

        for task in self.tasks:
            splits = task.get_splits(n_folds=self.config.n_splits, seed=self.config.seed)
            for model in self.models:
                for split in splits:
                    model.reset()
                    for mode in self.config.modes:
                        pbar.set_postfix(
                            model=model.info.name, task=task.name,
                            fold=split.fold, mode=mode.value,
                        )
                        cached = self._load_cached(model, task, split.fold, mode)
                        if cached is not None:
                            logger.info("Skipping cached: %s", cached.summary())
                            all_results.append(cached)
                            pbar.update()
                            continue

                        result = self._run_one(model, task, split, mode)
                        self._save_result(result)
                        all_results.append(result)
                        logger.info(result.summary())
                        pbar.update()

        pbar.close()
        return all_results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _run_one(
        self,
        model: ModelAdapter,
        task: Task,
        split,
        mode: EvalMode,
    ) -> TaskResult:
        if mode == EvalMode.ZERO_SHOT:
            return self._run_zero_shot(model, task, split)
        return self._run_fine_tuned(model, task, split)

    def _run_zero_shot(self, model: ModelAdapter, task: Task, split) -> TaskResult:
        emb_train = model.get_embeddings(split.adata_train)
        emb_test = model.get_embeddings(split.adata_test)

        targets_train = task.get_probe_targets(split.adata_train, split.label_key)
        targets_test = task.get_probe_targets(split.adata_test, split.label_key)

        predictions = task.run_linear_probe(emb_train, targets_train, emb_test)
        metrics = task.evaluate(predictions, targets_test)

        return TaskResult(
            model=model.info.name,
            task=task.name,
            fold=split.fold,
            mode=EvalMode.ZERO_SHOT,
            metrics=metrics,
            predictions=predictions,
            labels=targets_test,
        )

    def _run_fine_tuned(self, model: ModelAdapter, task: Task, split) -> TaskResult:
        if not model.info.supports_fine_tuning:
            logger.warning(
                "%s does not support fine-tuning — skipping fine-tuned eval for %s.",
                model.info.name, task.name,
            )
            return TaskResult(
                model=model.info.name,
                task=task.name,
                fold=split.fold,
                mode=EvalMode.FINE_TUNED,
                metrics={},
            )

        model.fine_tune(
            split.adata_train,
            task_type=task.name,
            label_key=split.label_key,
        )
        predictions = model.predict(split.adata_test, task_type=task.name)
        targets_test = task.get_probe_targets(split.adata_test, split.label_key)
        metrics = task.evaluate(predictions, targets_test)

        return TaskResult(
            model=model.info.name,
            task=task.name,
            fold=split.fold,
            mode=EvalMode.FINE_TUNED,
            metrics=metrics,
            predictions=predictions,
            labels=targets_test,
        )

    def _result_path(self, model: ModelAdapter, task: Task, fold: int, mode: EvalMode) -> Path:
        model_dir = self.config.output_dir / model.info.name
        model_dir.mkdir(parents=True, exist_ok=True)
        return model_dir / f"{task.name}_fold{fold}_{mode.value}.json"

    def _save_result(self, result: TaskResult) -> None:
        path = self.config.output_dir / result.model / f"{result.task}_fold{result.fold}_{result.mode.value}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": result.model,
            "task": result.task,
            "fold": result.fold,
            "mode": result.mode.value,
            "metrics": result.metrics,
        }
        path.write_text(json.dumps(payload, indent=2))

    def _load_cached(
        self, model: ModelAdapter, task: Task, fold: int, mode: EvalMode
    ) -> Optional[TaskResult]:
        path = self._result_path(model, task, fold, mode)
        if not path.exists():
            return None
        data = json.loads(path.read_text())
        return TaskResult(
            model=data["model"],
            task=data["task"],
            fold=data["fold"],
            mode=EvalMode(data["mode"]),
            metrics=data["metrics"],
        )
