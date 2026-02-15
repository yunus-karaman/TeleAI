from __future__ import annotations

from pathlib import Path

import pytest

from main import run_pipeline


def _write_config(config_path: Path, dataset_path: Path, artifacts_dir: Path, sample_size: int) -> None:
    config_path.write_text(
        f"""
paths:
  dataset: {dataset_path}
  artifacts_dir: {artifacts_dir}
  schema_report: {artifacts_dir / "schema_report.json"}
  clean_complaints: {artifacts_dir / "complaints_clean.jsonl"}
  preprocess_report: {artifacts_dir / "preprocess_report.json"}
  duplicates_report: {artifacts_dir / "duplicates_report.json"}
  quarantine: {artifacts_dir / "quarantine.jsonl"}
  logs_dir: {artifacts_dir / "logs"}
reproducibility:
  seed: 42
  deterministic: true
logging:
  level: INFO
  file: {artifacts_dir / "logs" / "pipeline.jsonl"}
pipeline:
  fail_fast_stage: true
  continue_on_record_error: true
  max_retry_count: 1
preprocess:
  version: preprocess-test
  output_timestamp_iso: "2026-01-01T00:00:00+00:00"
  min_chars: 80
  max_chars: 6000
  script_noise:
    indicators: ["<script", "googletag", "prebid", "pubads", "window.addEventListener"]
    min_indicator_hits: 2
    min_js_line_ratio: 0.35
    min_alpha_ratio: 0.25
    min_cleaned_ratio: 0.2
  multi_complaint:
    strategy: truncate_primary
    split_min_chars: 80
  duplicates:
    enabled: true
    shingle_size: 3
    num_perm: 16
    bands: 4
    similarity_threshold: 0.9
    full_mode_drop_duplicates: true
    smoke_mode_keep_duplicates: true
    smoke_drop_extreme_clusters: false
    smoke_extreme_cluster_size: 20
mode_profiles:
  SMOKE:
    mode_runtime:
      sample_size: {sample_size}
  FULL:
    mode_runtime:
      sample_size: null
""".strip(),
        encoding="utf-8",
    )


def test_preprocess_smoke_on_real_subset(tmp_path: Path) -> None:
    source_dataset = Path("sikayetler.jsonl")
    if not source_dataset.exists():
        pytest.skip("sikayetler.jsonl not found in repository root.")

    subset_dataset = tmp_path / "subset.jsonl"
    lines = source_dataset.read_text(encoding="utf-8").splitlines()[:80]
    subset_dataset.write_text("\n".join(lines), encoding="utf-8")

    artifacts_dir = tmp_path / "artifacts"
    config_path = tmp_path / "config.yaml"
    _write_config(config_path, subset_dataset, artifacts_dir, sample_size=80)

    exit_code = run_pipeline(mode="SMOKE", config_path=str(config_path), stage="preprocess")
    assert exit_code == 0

    clean_path = artifacts_dir / "complaints_clean.jsonl"
    report_path = artifacts_dir / "preprocess_report.json"
    quarantine_path = artifacts_dir / "quarantine.jsonl"
    assert clean_path.exists()
    assert report_path.exists()
    assert quarantine_path.exists()
    assert clean_path.read_text(encoding="utf-8").strip() != ""

