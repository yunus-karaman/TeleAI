from __future__ import annotations

import hashlib
import json
from pathlib import Path

from main import run_pipeline


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _normalized_record(record_id: str, category: str, text: str, created_at: str) -> dict:
    return {
        "schema_name": "NormalizedComplaint",
        "schema_version": "1.0.0",
        "schema_revision": 1,
        "complaint_id": record_id,
        "brand_name": "DemoTel",
        "brand_slug": "demotel",
        "created_at_iso": created_at,
        "title_clean": "Test Baslik",
        "complaint_text_clean": text,
        "normalized_category": category,
        "confidence_score": 0.8,
        "assignment_reason": "test assignment",
        "needs_review": False,
        "source_category": category,
        "quality_flags": [],
        "duplicate_cluster_id": None,
        "is_duplicate_of": None,
        "taxonomy_version": "1.0.0",
        "source_hash_sha256": _sha(record_id + text),
    }


def test_solution_steps_stage_smoke_generates_artifacts(tmp_path: Path) -> None:
    raw_dataset = tmp_path / "raw.jsonl"
    raw_dataset.write_text(
        json.dumps(
            {
                "brand_name": "DemoTel",
                "brand_slug": "demotel",
                "title": "ornek kayit",
                "complaint_text": "Bu metin schema analiz asamasini gecmek icin yeterli uzunluktadir.",
                "normalized_category": "INTERNET",
                "tags": [],
                "support_count": 0,
                "is_synthetic": False,
                "quality_flags": [],
            }
        ),
        encoding="utf-8",
    )

    categories = ["BILLING_PAYMENTS", "MOBILE_DATA_SPEED", "MOBILE_VOICE_SMS"]
    all_records = []
    for index in range(12):
        category = categories[index % len(categories)]
        text = (
            "Fatura odeme sorunu var."
            if category == "BILLING_PAYMENTS"
            else "Mobil internet hizi dusuk ve baglanti kopuyor."
            if category == "MOBILE_DATA_SPEED"
            else "Arama ve SMS hizmeti tutarsiz calisiyor."
        )
        all_records.append(
            _normalized_record(
                record_id=f"n-{index:03d}",
                category=category,
                text=text + f" Ornek {index}.",
                created_at=f"2025-01-{(index % 28) + 1:02d}T10:00:00",
            )
        )

    artifacts_dir = tmp_path / "artifacts"
    splits_dir = artifacts_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    for split_name in ["train", "val", "test", "hard_test"]:
        target = splits_dir / f"{split_name}.jsonl"
        target.write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in all_records), encoding="utf-8")

    labeled_path = artifacts_dir / "complaints_labeled.jsonl"
    labeled_path.write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in all_records), encoding="utf-8")

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
paths:
  dataset: {raw_dataset}
  artifacts_dir: {artifacts_dir}
  schema_report: {artifacts_dir / "schema_report.json"}
  clean_complaints: {artifacts_dir / "complaints_clean.jsonl"}
  labeled_complaints: {labeled_path}
  preprocess_report: {artifacts_dir / "preprocess_report.json"}
  duplicates_report: {artifacts_dir / "duplicates_report.json"}
  train_split: {splits_dir / "train.jsonl"}
  val_split: {splits_dir / "val.jsonl"}
  test_split: {splits_dir / "test.jsonl"}
  hard_test_split: {splits_dir / "hard_test.jsonl"}
  taxonomy_report_json: {artifacts_dir / "taxonomy_report.json"}
  taxonomy_report_md: {artifacts_dir / "taxonomy_report.md"}
  taxonomy_error_analysis_dir: {artifacts_dir / "error_analysis"}
  category_patterns: {artifacts_dir / "category_patterns.json"}
  solution_steps_jsonl: {artifacts_dir / "solution_steps.jsonl"}
  kb_jsonl: {artifacts_dir / "kb.jsonl"}
  step_kb_links_jsonl: {artifacts_dir / "step_kb_links.jsonl"}
  solution_steps_summary: {artifacts_dir / "solution_steps_summary.json"}
  solution_step_lint_report: {artifacts_dir / "solution_step_lint_report.json"}
  kb_lint_report: {artifacts_dir / "kb_lint_report.json"}
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
solution_steps:
  version: solution-steps-v1
  taxonomy_file: taxonomy/taxonomy.yaml
  pattern_top_k: 5
  smoke_category_limit: 3
  quality:
    min_steps_per_category: 6
    max_steps_per_category: 12
    min_level_counts:
      L1: 3
      L2: 2
      L3: 1
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

    exit_code = run_pipeline(mode="SMOKE", config_path=str(config_path), stage="solution_steps")
    assert exit_code == 0

    assert (artifacts_dir / "category_patterns.json").exists()
    assert (artifacts_dir / "solution_steps.jsonl").exists()
    assert (artifacts_dir / "kb.jsonl").exists()
    assert (artifacts_dir / "step_kb_links.jsonl").exists()
    assert (artifacts_dir / "solution_steps_summary.json").exists()
    assert (artifacts_dir / "solution_step_lint_report.json").exists()
    assert (artifacts_dir / "kb_lint_report.json").exists()

    summary = json.loads((artifacts_dir / "solution_steps_summary.json").read_text(encoding="utf-8"))
    assert summary["safety_lint_violations_count"] == 0
    assert sum(summary["count_per_category"].values()) == 18

