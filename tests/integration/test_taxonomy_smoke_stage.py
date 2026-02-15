from __future__ import annotations

import hashlib
import json
from pathlib import Path

from main import run_pipeline


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _build_clean_record(complaint_id: str, text: str, created_at: str, source_category: str) -> dict:
    return {
        "schema_name": "CleanComplaint",
        "schema_version": "1.0.0",
        "schema_revision": 1,
        "complaint_id": complaint_id,
        "brand_name": "DemoTel",
        "brand_slug": "demotel",
        "created_at_iso": created_at,
        "normalized_category": source_category,
        "original_category_label": source_category,
        "title_clean": "Test Baslik",
        "complaint_text_clean": text,
        "tags": [],
        "support_count": 1,
        "quality_flags": [],
        "preprocess_version": "preprocess-test",
        "preprocess_timestamp_iso": "2026-01-01T00:00:00+00:00",
        "source_hash_sha256": _sha(complaint_id + text),
        "duplicate_cluster_id": None,
        "is_duplicate_of": None,
    }


def test_taxonomy_stage_smoke_runs_and_produces_outputs(tmp_path: Path) -> None:
    dataset = tmp_path / "raw.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "brand_name": "DemoTel",
                "brand_slug": "demotel",
                "title": "ornek sikayet",
                "complaint_text": "Bu test kaydidir ve schema analiz adimini calistirmak icin yeterli uzunluktadir.",
                "normalized_category": "INTERNET",
                "tags": [],
                "support_count": 0,
                "is_synthetic": False,
                "quality_flags": [],
            }
        ),
        encoding="utf-8",
    )

    clean_path = tmp_path / "complaints_clean.jsonl"
    records = []
    for index in range(20):
        if index % 4 == 0:
            text = "Faturam odendigim halde borc gorunuyor ve odeme yansimiyor."
            source = "FATURA_ODEME"
        elif index % 4 == 1:
            text = "Yurt disinda roaming paketi aldim ama internet calismiyor."
            source = "ROAMING_YURTDISI"
        elif index % 4 == 2:
            text = "Numara tasima basvurum reddedildi ve surekli hata aliyorum."
            source = "NUMARA_TASIMA"
        else:
            text = "Musteri hizmetleri kayit acmiyor ve sureci yonetmiyor."
            source = "MUSTERI_HIZMETLERI"
        records.append(
            _build_clean_record(
                complaint_id=f"tc-{index:03d}",
                text=text + f" Detay {index}.",
                created_at=f"2025-01-{(index % 28) + 1:02d}T10:00:00",
                source_category=source,
            )
        )
    clean_path.write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in records), encoding="utf-8")

    artifacts_dir = tmp_path / "artifacts"
    config = tmp_path / "config.yaml"
    config.write_text(
        f"""
paths:
  dataset: {dataset}
  artifacts_dir: {artifacts_dir}
  schema_report: {artifacts_dir / "schema_report.json"}
  clean_complaints: {clean_path}
  labeled_complaints: {artifacts_dir / "complaints_labeled.jsonl"}
  preprocess_report: {artifacts_dir / "preprocess_report.json"}
  duplicates_report: {artifacts_dir / "duplicates_report.json"}
  train_split: {artifacts_dir / "splits" / "train.jsonl"}
  val_split: {artifacts_dir / "splits" / "val.jsonl"}
  test_split: {artifacts_dir / "splits" / "test.jsonl"}
  hard_test_split: {artifacts_dir / "splits" / "hard_test.jsonl"}
  taxonomy_report_json: {artifacts_dir / "taxonomy_report.json"}
  taxonomy_report_md: {artifacts_dir / "taxonomy_report.md"}
  taxonomy_error_analysis_dir: {artifacts_dir / "error_analysis"}
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
taxonomy:
  taxonomy_file: taxonomy/taxonomy.yaml
  assignment:
    min_confidence: 0.55
    low_confidence_policy: other
    review_margin_threshold: 0.08
    rule_weight: 0.55
    embedding_weight: 0.45
    keyword_weight: 1.0
    negative_weight: 0.8
    example_weight: 1.2
    embedding:
      max_features: 3000
      ngram_min: 1
      ngram_max: 2
      min_df: 1
  splits:
    train_ratio: 0.7
    val_ratio: 0.1
    test_ratio: 0.2
    hard_test_ratio: 0.1
    hard_test_max_size: 100
    hard_short_max_chars: 180
    hard_confidence_threshold: 0.65
  baselines:
    run_baseline2_smoke: false
    run_baseline2_full: true
    baseline1:
      min_df: 1
      max_features: 5000
      max_iter: 250
    baseline2:
      min_df: 1
      max_features: 5000
      svd_components: 64
      max_iter: 300
  error_analysis:
    top_n_per_class: 10
  report:
    needs_review_sample_size: 5
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

    exit_code = run_pipeline(mode="SMOKE", config_path=str(config), stage="taxonomy")
    assert exit_code == 0

    assert (artifacts_dir / "complaints_labeled.jsonl").exists()
    assert (artifacts_dir / "splits" / "train.jsonl").exists()
    assert (artifacts_dir / "splits" / "val.jsonl").exists()
    assert (artifacts_dir / "splits" / "test.jsonl").exists()
    assert (artifacts_dir / "splits" / "hard_test.jsonl").exists()
    assert (artifacts_dir / "taxonomy_report.json").exists()
    assert (artifacts_dir / "taxonomy_report.md").exists()

    report = json.loads((artifacts_dir / "taxonomy_report.json").read_text(encoding="utf-8"))
    assert "baseline_tfidf_linear" in report["baselines"]

