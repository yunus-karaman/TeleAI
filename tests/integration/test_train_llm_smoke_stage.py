from __future__ import annotations

import hashlib
import json
from pathlib import Path

from main import run_pipeline


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows), encoding="utf-8")


def _normalized_record(record_id: str, text: str, created_at: str) -> dict:
    return {
        "schema_name": "NormalizedComplaint",
        "schema_version": "1.0.0",
        "schema_revision": 1,
        "complaint_id": record_id,
        "brand_name": "DemoTel",
        "brand_slug": "demotel",
        "created_at_iso": created_at,
        "title_clean": "test baslik",
        "complaint_text_clean": text,
        "normalized_category": "OTHER",
        "confidence_score": 0.8,
        "assignment_reason": "test assignment",
        "needs_review": False,
        "source_category": "OTHER",
        "quality_flags": [],
        "duplicate_cluster_id": None,
        "is_duplicate_of": None,
        "taxonomy_version": "1.0.0",
        "source_hash_sha256": _sha(record_id + text),
    }


def _solution_step_record(step_id: str, level: str, title: str) -> dict:
    return {
        "schema_name": "SolutionStep",
        "schema_version": "1.0.0",
        "schema_revision": 1,
        "step_id": step_id,
        "category_id": "OTHER",
        "level": level,
        "title_tr": title,
        "instructions_tr": [
            "Belirtiyi net olarak kaydedin ve tekrari not edin.",
            "Ayni testi farkli ag veya zaman diliminde tekrar edin.",
            "Hata mesaji varsa kayit altina alin.",
        ],
        "required_inputs": ["device_os", "error_code", "city_region"],
        "success_check": "Sorunun kapsam ve tekrar kosullari netlesir.",
        "stop_conditions": ["Iki testte de devam ederse resmi destek kanalina aktar."],
        "escalation_unit": "GENERAL_SUPPORT",
        "risk_level": "low" if level == "L1" else "medium",
        "tags": ["other", "smoke_train"],
        "version": "solution-steps-v1",
    }


def _kb_record(doc_id: str, paragraph_id: str, step_id: str, confidence: float) -> dict:
    return {
        "schema_name": "KBParagraph",
        "schema_version": "1.0.0",
        "schema_revision": 1,
        "doc_id": doc_id,
        "paragraph_id": paragraph_id,
        "text_tr": "Bu kanit metni ilgili adimin neden gerekli oldugunu guvenli ve genel sekilde aciklar.",
        "applies_to_step_ids": [step_id],
        "source_type": "internal_best_practice",
        "confidence": confidence,
        "version": "solution-steps-v1",
    }


def test_train_llm_smoke_stage_completes_and_writes_artifacts(tmp_path: Path) -> None:
    raw_dataset = tmp_path / "raw.jsonl"
    raw_dataset.write_text(
        json.dumps(
            {
                "brand_name": "DemoTel",
                "brand_slug": "demotel",
                "title": "test kaydi",
                "complaint_text": "Bu satir schema analiz adimini calistirmak icin yeterince uzundur ve gecerlidir.",
                "normalized_category": "OTHER",
                "tags": [],
                "support_count": 0,
                "is_synthetic": False,
                "quality_flags": [],
            }
        ),
        encoding="utf-8",
    )

    artifacts_dir = tmp_path / "artifacts"
    splits_dir = artifacts_dir / "splits"

    records = []
    for index in range(18):
        records.append(
            _normalized_record(
                record_id=f"t-{index:03d}",
                text=f"Hizmette tekrarli baglanti sorunu yasaniyor ve kullanimi etkiliyor. Ornek {index}.",
                created_at=f"2025-03-{(index % 27) + 1:02d}T12:00:00+00:00",
            )
        )

    _write_jsonl(artifacts_dir / "complaints_labeled.jsonl", records)
    _write_jsonl(splits_dir / "train.jsonl", records[:12])
    _write_jsonl(splits_dir / "val.jsonl", records[12:15])
    _write_jsonl(splits_dir / "test.jsonl", records[15:17])
    _write_jsonl(splits_dir / "hard_test.jsonl", records[17:18])

    _write_jsonl(
        artifacts_dir / "solution_steps.jsonl",
        [
            _solution_step_record("STEP.OTHER.001", "L1", "Genel tespit"),
            _solution_step_record("STEP.OTHER.002", "L1", "Baglanti dogrulama"),
            _solution_step_record("STEP.OTHER.003", "L2", "Eskalasyon hazirligi"),
        ],
    )
    _write_jsonl(
        artifacts_dir / "kb.jsonl",
        [
            _kb_record("KB.OTHER.001", "KB.OTHER.001#P1", "STEP.OTHER.001", 0.82),
            _kb_record("KB.OTHER.002", "KB.OTHER.002#P1", "STEP.OTHER.002", 0.80),
            _kb_record("KB.OTHER.003", "KB.OTHER.003#P1", "STEP.OTHER.003", 0.78),
        ],
    )
    _write_jsonl(
        artifacts_dir / "step_kb_links.jsonl",
        [
            {
                "step_id": "STEP.OTHER.001",
                "evidence_ids": ["KB.OTHER.001#P1"],
                "rationale": "Temel tespit adimini destekler.",
                "version": "solution-steps-v1",
            },
            {
                "step_id": "STEP.OTHER.002",
                "evidence_ids": ["KB.OTHER.002#P1"],
                "rationale": "Dogrulama adimini destekler.",
                "version": "solution-steps-v1",
            },
            {
                "step_id": "STEP.OTHER.003",
                "evidence_ids": ["KB.OTHER.003#P1"],
                "rationale": "Eskalasyon adimini destekler.",
                "version": "solution-steps-v1",
            },
        ],
    )

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
paths:
  dataset: {raw_dataset}
  artifacts_dir: {artifacts_dir}
  schema_report: {artifacts_dir / "schema_report.json"}
  labeled_complaints: {artifacts_dir / "complaints_labeled.jsonl"}
  train_split: {splits_dir / "train.jsonl"}
  val_split: {splits_dir / "val.jsonl"}
  test_split: {splits_dir / "test.jsonl"}
  hard_test_split: {splits_dir / "hard_test.jsonl"}
  solution_steps_jsonl: {artifacts_dir / "solution_steps.jsonl"}
  kb_jsonl: {artifacts_dir / "kb.jsonl"}
  step_kb_links_jsonl: {artifacts_dir / "step_kb_links.jsonl"}
  embeddings_dir: {artifacts_dir / "embeddings"}
  task2_sft_train: {artifacts_dir / "training" / "task2_sft_train.jsonl"}
  task2_sft_val: {artifacts_dir / "training" / "task2_sft_val.jsonl"}
  task1_intent_train: {artifacts_dir / "training" / "task1_intent_train.jsonl"}
  task1_intent_val: {artifacts_dir / "training" / "task1_intent_val.jsonl"}
  models_dir: {artifacts_dir / "models"}
  training_eval_quick_json: {artifacts_dir / "training_eval_quick.json"}
  training_eval_quick_md: {artifacts_dir / "training_eval_quick.md"}
  training_run_summary: {artifacts_dir / "training_run_summary.json"}
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
graph_pipeline:
  include_brand_nodes: true
  include_time_bucket_nodes: true
  embeddings:
    dimension: 96
    ngram_min: 1
    ngram_max: 2
    force_recompute: true
  retrieval:
    alpha: 0.45
    beta: 0.15
    gamma: 0.25
    delta: 0.10
    lambda_gnn: 0.0
    min_steps: 3
    top_steps: 5
    max_evidence: 10
    escalation_threshold: 0.58
model:
  base_model_name: Qwen/Qwen2.5-7B-Instruct
  fallback_model_name: mistralai/Mistral-7B-Instruct-v0.3
  cache_dir: {artifacts_dir / "hf_cache"}
  local_files_only: true
training:
  use_llm_for_intent: false
training_llm:
  dataset:
    version: task2-sft-v1
    task2_top_steps: 5
    task2_min_steps: 3
    task2_max_evidence: 10
    force_recompute_retrieval_embeddings: true
  smoke:
    retrieval_fit_limit: 100
    task2_train_limit: 12
    task2_val_limit: 3
  lora:
    r: 16
    alpha: 16
    dropout: 0.05
    target_modules: [q_proj, k_proj, v_proj, o_proj]
    use_qlora_4bit: true
  trainer:
    force_mock_training: true
    fallback_to_mock_on_failure: true
    SMOKE:
      learning_rate: 2.0e-4
      weight_decay: 0.01
      warmup_ratio: 0.03
      gradient_accumulation_steps: 2
      max_seq_len: 1024
      num_epochs: 1
      batch_size: 2
      eval_steps: 10
      save_steps: 10
      logging_steps: 5
      max_oom_retries: 1
      auto_batch_reduce_on_oom: true
      early_stopping_patience: 2
    FULL:
      learning_rate: 2.0e-4
      weight_decay: 0.01
      warmup_ratio: 0.03
      gradient_accumulation_steps: 2
      max_seq_len: 1024
      num_epochs: 1
      batch_size: 2
      eval_steps: 10
      save_steps: 10
      logging_steps: 5
      max_oom_retries: 1
      auto_batch_reduce_on_oom: true
      early_stopping_patience: 2
  inference:
    default_run_id: null
    force_renderer_only: true
    max_new_tokens: 300
    repetition_penalty: 1.1
    regen_once_on_invalid: true
  quick_eval:
    SMOKE:
      test_limit: 2
      hard_test_limit: 1
    FULL:
      test_limit: 2
      hard_test_limit: 1
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

    exit_code = run_pipeline(mode="SMOKE", config_path=str(config_path), stage="train_llm")
    assert exit_code == 0

    assert (artifacts_dir / "training" / "task2_sft_train.jsonl").exists()
    assert (artifacts_dir / "training" / "task2_sft_val.jsonl").exists()
    assert (artifacts_dir / "training_eval_quick.json").exists()
    assert (artifacts_dir / "training_eval_quick.md").exists()
    assert (artifacts_dir / "training_run_summary.json").exists()

    model_dirs = [path for path in (artifacts_dir / "models").glob("*") if path.is_dir()]
    assert len(model_dirs) >= 1
    assert (model_dirs[0] / "adapter_config.json").exists()
