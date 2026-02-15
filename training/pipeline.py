from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from scripts.logging_utils import log_event
from training.data_builder import build_and_write_training_datasets
from training.lora_trainer import run_lora_training
from training.quick_eval import run_training_quick_eval


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def run_train_llm_stage(*, config: dict[str, Any], mode: str, logger: logging.Logger) -> dict[str, Any]:
    dataset_stats = build_and_write_training_datasets(config=config, mode=mode, logger=logger)
    train_result = run_lora_training(config=config, mode=mode, logger=logger)
    quick_eval = run_training_quick_eval(config=config, mode=mode, logger=logger, run_id=train_result.run_id)

    summary = {
        "mode": mode,
        "dataset": dataset_stats,
        "training": {
            "run_id": train_result.run_id,
            "run_dir": str(train_result.run_dir),
            "selected_model": train_result.selected_model,
            "backend": train_result.backend,
            "status": train_result.status,
            "metrics": train_result.metrics,
        },
        "quick_eval": quick_eval,
    }
    _write_json(Path(config["paths"]["training_run_summary"]), summary)
    log_event(
        logger,
        "INFO",
        "train_llm_stage_complete",
        {
            "run_id": train_result.run_id,
            "model": train_result.selected_model,
            "backend": train_result.backend,
            "train_examples": dataset_stats["task2_train"],
            "val_examples": dataset_stats["task2_val"],
            "quick_eval_path": config["paths"]["training_eval_quick_json"],
        },
    )
    return summary
