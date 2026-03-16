from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional


class ExperimentLogger:
    def __init__(
        self,
        *,
        save_dir: Path,
        use_wandb: bool,
        project: str,
        group: str,
        name: str,
        config: Dict,
    ) -> None:
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.save_dir / "metrics.jsonl"
        self.stdout_path = self.save_dir / "stdout.log"
        (self.save_dir / "config.json").write_text(
            json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        self._wandb_run: Optional[object] = None

        if use_wandb:
            try:
                import wandb  # type: ignore

                self._wandb_run = wandb.init(
                    project=project,
                    group=group,
                    name=name,
                    config=config,
                    dir=str(self.save_dir),
                )
            except Exception:
                self._wandb_run = None

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        record = dict(metrics)
        if step is not None:
            record["step"] = step
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        if self._wandb_run is not None:
            self._wandb_run.log(metrics, step=step)

    def write_eval(self, metrics: Dict[str, float]) -> None:
        (self.save_dir / "eval.json").write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        if self._wandb_run is not None:
            self._wandb_run.summary.update(metrics)

    def finish(self) -> None:
        if self._wandb_run is not None:
            self._wandb_run.finish()

    def log_text(self, message: str) -> None:
        with self.stdout_path.open("a", encoding="utf-8") as handle:
            handle.write(message + "\n")
