from __future__ import annotations

from pathlib import Path

import pytest

from scripts.config_loader import load_config


def test_load_config_with_mode_merge(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        """
logging:
  level: INFO
  file: artifacts/logs/pipeline.jsonl
mode_profiles:
  SMOKE:
    mode_runtime:
      sample_size: 10
  FULL:
    mode_runtime:
      sample_size: null
""".strip(),
        encoding="utf-8",
    )

    config = load_config(config_file, "SMOKE")
    assert config["mode_runtime"]["sample_size"] == 10
    assert config["runtime"]["mode"] == "SMOKE"


def test_full_mode_allows_mock_fallback_enabled(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        """
training_llm:
  trainer:
    fallback_to_mock_on_failure: true
mode_profiles:
  FULL:
    mode_runtime:
      sample_size: null
  SMOKE:
    mode_runtime:
      sample_size: 10
""".strip(),
        encoding="utf-8",
    )

    config = load_config(config_file, "FULL")
    assert config["training_llm"]["trainer"]["fallback_to_mock_on_failure"] is True


def test_full_mode_rejects_force_mock_training(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        """
training_llm:
  trainer:
    fallback_to_mock_on_failure: false
    force_mock_training: true
mode_profiles:
  FULL:
    mode_runtime:
      sample_size: null
  SMOKE:
    mode_runtime:
      sample_size: 10
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="force_mock_training"):
        load_config(config_file, "FULL")
