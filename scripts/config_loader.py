from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def _deep_merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge_dicts(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _validate_mode_invariants(config: dict[str, Any], mode: str) -> None:
    if mode != "FULL":
        return
    trainer_cfg = config.get("training_llm", {}).get("trainer", {})
    if bool(trainer_cfg.get("force_mock_training", False)):
        raise ValueError("FULL mode invariant violated: training_llm.trainer.force_mock_training must be false.")


def load_config(config_path: str | Path, mode: str) -> dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Config must be a top-level mapping.")

    profiles = raw.get("mode_profiles", {})
    if mode not in profiles:
        raise ValueError(f"Mode '{mode}' not found in mode_profiles.")

    merged = _deep_merge_dicts(raw, profiles[mode])
    _validate_mode_invariants(merged, mode)
    merged["runtime"] = {"mode": mode}
    return merged
