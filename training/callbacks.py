from __future__ import annotations

import json
from pathlib import Path

from lightning.pytorch import Callback
from lightning.pytorch.callbacks import TQDMProgressBar


class CleanProgressBar(TQDMProgressBar):
    """Keep tqdm's step/elapsed/remaining display and hide loss, it/s and v_num."""

    def get_metrics(self, trainer, model):
        return {}

    def init_train_tqdm(self):
        bar = super().init_train_tqdm()
        bar.bar_format = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
        return bar

    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm()
        bar.bar_format = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
        return bar


class BestArtifactSaver(Callback):
    def __init__(self, output_dir: str | Path, metrics=None) -> None:
        self.output_dir = Path(output_dir)
        self.best = {name: float("-inf") for name in (metrics or ("bleu4", "meteor"))}

    def on_validation_end(self, trainer, pl_module) -> None:
        if not trainer.is_global_zero:
            return
        metrics = pl_module.latest_valid_metrics
        for name in self.best:
            if (
                name in {"clinical_checkpoint_score", "clinical_score_v2"}
                and metrics.get("checkpoint_hard_gate_pass", 1.0) < 0.5
            ):
                continue
            value = metrics.get(name)
            if value is None or value <= self.best[name]:
                continue
            self.best[name] = value
            artifact_name = (
                "clinical_v2"
                if name == "clinical_score_v2"
                else ("clinical" if name == "clinical_checkpoint_score" else name)
            )
            destination = self.output_dir / f"best_{artifact_name}"
            pl_module.save_components(destination)
            metadata = {
                "metric": name,
                "value": value,
                "epoch": int(pl_module.current_epoch) + 1,
                "pretrained_model": pl_module.cfg["Model"]["pretrained_path"],
                "metric_schema_version": 2 if name == "clinical_score_v2" else 1,
                "all_metrics": metrics,
            }
            destination.mkdir(parents=True, exist_ok=True)
            (destination / "metadata.json").write_text(
                json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
