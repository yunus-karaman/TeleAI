from __future__ import annotations

import hashlib
import json
from pathlib import Path

from main import run_pipeline


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_config(config_path: Path, dataset_path: Path, artifacts_dir: Path) -> None:
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
  seed: 123
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
    similarity_threshold: 0.85
    full_mode_drop_duplicates: true
    smoke_mode_keep_duplicates: true
    smoke_drop_extreme_clusters: false
    smoke_extreme_cluster_size: 20
mode_profiles:
  SMOKE:
    mode_runtime:
      sample_size: null
  FULL:
    mode_runtime:
      sample_size: null
""".strip(),
        encoding="utf-8",
    )


def test_preprocess_determinism_same_seed_same_output(tmp_path: Path) -> None:
    dataset = tmp_path / "determinism.jsonl"
    records = [
        {
            "complaint_id": "1",
            "url": "https://example.com/1",
            "brand_name": "DemoTel",
            "brand_slug": "demotel",
            "title": "Baglanti sorunu",
            "complaint_text": "0532 111 22 33 numarali hattimda internet surekli kopuyor ve cozum bekliyorum.",
            "created_at_iso": "2026-01-01T10:00:00",
            "normalized_category": "INTERNET",
            "original_category_label": "Kesinti",
            "tags": ["internet"],
            "support_count": 1,
            "is_synthetic": False,
            "quality_flags": [],
        },
        {
            "complaint_id": "2",
            "url": "https://example.com/2",
            "brand_name": "DemoTel",
            "brand_slug": "demotel",
            "title": "Baglanti sorunu ikinci kayit",
            "complaint_text": "0532 111 22 33 numarali hattimda internet surekli kopuyor ve cozum bekliyorum.",
            "created_at_iso": "2026-01-02T10:00:00",
            "normalized_category": "INTERNET",
            "original_category_label": "Kesinti",
            "tags": ["internet"],
            "support_count": 1,
            "is_synthetic": False,
            "quality_flags": [],
        },
    ]
    dataset.write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in records), encoding="utf-8")

    artifacts_a = tmp_path / "a"
    artifacts_b = tmp_path / "b"
    config_a = tmp_path / "config_a.yaml"
    config_b = tmp_path / "config_b.yaml"
    _write_config(config_a, dataset, artifacts_a)
    _write_config(config_b, dataset, artifacts_b)

    assert run_pipeline(mode="SMOKE", config_path=str(config_a), stage="preprocess") == 0
    assert run_pipeline(mode="SMOKE", config_path=str(config_b), stage="preprocess") == 0

    assert _sha256_file(artifacts_a / "complaints_clean.jsonl") == _sha256_file(artifacts_b / "complaints_clean.jsonl")
    assert _sha256_file(artifacts_a / "duplicates_report.json") == _sha256_file(artifacts_b / "duplicates_report.json")
    assert _sha256_file(artifacts_a / "preprocess_report.json") == _sha256_file(artifacts_b / "preprocess_report.json")

