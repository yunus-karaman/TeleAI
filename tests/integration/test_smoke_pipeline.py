from __future__ import annotations

import json
from pathlib import Path

from main import run_pipeline


def test_smoke_preprocess_pipeline_end_to_end(tmp_path: Path) -> None:
    dataset = tmp_path / "sample.jsonl"
    dataset.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "complaint_id": "1",
                        "url": "https://example.com/1",
                        "brand_name": "DemoTel",
                        "brand_slug": "demotel",
                        "title": "Baglanti sorunu",
                        "complaint_text": "Hat cekiyor gibi gorunuyor ama internet baglantisi surekli kopuyor ve stabil degil.",
                        "created_at_iso": "2026-01-01T12:00:00",
                        "normalized_category": "INTERNET",
                        "original_category_label": "Kesinti",
                        "tags": ["internet"],
                        "support_count": 1,
                        "is_synthetic": False,
                        "quality_flags": [],
                        "scraped_at_iso": "2026-01-02T12:00:00+00:00",
                        "http_status": 200,
                        "parse_version": "v1",
                    }
                ),
                "{malformed_json}",
            ]
        ),
        encoding="utf-8",
    )

    config = tmp_path / "config.yaml"
    config.write_text(
        f"""
paths:
  dataset: {dataset}
  artifacts_dir: {tmp_path / "artifacts"}
  schema_report: {tmp_path / "artifacts" / "schema_report.json"}
  clean_complaints: {tmp_path / "artifacts" / "complaints_clean.jsonl"}
  preprocess_report: {tmp_path / "artifacts" / "preprocess_report.json"}
  duplicates_report: {tmp_path / "artifacts" / "duplicates_report.json"}
  quarantine: {tmp_path / "artifacts" / "quarantine.jsonl"}
  logs_dir: {tmp_path / "artifacts" / "logs"}
reproducibility:
  seed: 42
  deterministic: true
logging:
  level: INFO
  file: {tmp_path / "artifacts" / "logs" / "pipeline.jsonl"}
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
      sample_size: 10
  FULL:
    mode_runtime:
      sample_size: null
""".strip(),
        encoding="utf-8",
    )

    exit_code = run_pipeline(mode="SMOKE", config_path=str(config), stage="preprocess")
    assert exit_code == 0
    assert (tmp_path / "artifacts" / "complaints_clean.jsonl").exists()
    assert (tmp_path / "artifacts" / "preprocess_report.json").exists()
    assert (tmp_path / "artifacts" / "duplicates_report.json").exists()
    assert (tmp_path / "artifacts" / "quarantine.jsonl").exists()
