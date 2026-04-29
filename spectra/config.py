from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

from spectra.runner import RunConfig


@dataclass
class OutputConfig:
    dir: str = "./results"
    format: list[str] = field(default_factory=lambda: ["json", "csv"])


@dataclass
class Settings:
    zero_shot: bool = True
    fine_tuned: bool = True
    n_splits: int = 5
    seed: int = 42


@dataclass
class SpectraConfig:
    models: list[str]
    tasks: list[str]
    settings: Settings
    output: OutputConfig

    def to_run_config(self) -> RunConfig:
        return RunConfig(
            output_dir=Path(self.output.dir),
            n_splits=self.settings.n_splits,
            seed=self.settings.seed,
            zero_shot=self.settings.zero_shot,
            fine_tuned=self.settings.fine_tuned,
        )


def load_config(path: str | Path) -> SpectraConfig:
    raw = yaml.safe_load(Path(path).read_text())

    settings_raw = raw.get("settings", {})
    settings = Settings(
        zero_shot=settings_raw.get("zero_shot", True),
        fine_tuned=settings_raw.get("fine_tuned", True),
        n_splits=settings_raw.get("n_splits", 5),
        seed=settings_raw.get("seed", 42),
    )

    output_raw = raw.get("output", {})
    output = OutputConfig(
        dir=output_raw.get("dir", "./results"),
        format=output_raw.get("format", ["json", "csv"]),
    )

    return SpectraConfig(
        models=raw["models"],
        tasks=raw["tasks"],
        settings=settings,
        output=output,
    )
