from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

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


def _solution_step_record(step_id: str, level: str, title: str, escalation_unit: str) -> dict:
    return {
        "schema_name": "SolutionStep",
        "schema_version": "1.0.0",
        "schema_revision": 1,
        "step_id": step_id,
        "category_id": "OTHER",
        "level": level,
        "title_tr": title,
        "instructions_tr": [
            "Sorunun tekrar ettigi zamani net bicimde not edin.",
            "Ayni testi farkli bir baglanti ortaminda tekrar edin.",
            "Hata mesaji varsa ekran uyarisini kaydedin.",
        ],
        "required_inputs": ["device_os", "error_code", "city_region"],
        "success_check": "Sorunun kapsam ve tekrar kosullari netlesir.",
        "stop_conditions": ["Iki denemede de duzelmezse resmi destek ekibine aktar."],
        "escalation_unit": escalation_unit,
        "risk_level": "low" if level == "L1" else "medium",
        "tags": ["other", "smoke_test"],
        "version": "solution-steps-v1",
    }


def _kb_record(doc_id: str, paragraph_id: str, step_id: str, confidence: float) -> dict:
    return {
        "schema_name": "KBParagraph",
        "schema_version": "1.0.0",
        "schema_revision": 1,
        "doc_id": doc_id,
        "paragraph_id": paragraph_id,
        "text_tr": "Bu kanit metni adimin neden uygulandigini genel ve guvenli bicimde aciklar.",
        "applies_to_step_ids": [step_id],
        "source_type": "internal_best_practice",
        "confidence": confidence,
        "version": "solution-steps-v1",
    }


def _build_stage_graph_fixture(tmp_path: Path, use_gnn: bool) -> tuple[Path, Path]:
    raw_dataset = tmp_path / "raw.jsonl"
    raw_dataset.write_text(
        json.dumps(
            {
                "brand_name": "DemoTel",
                "brand_slug": "demotel",
                "title": "test kaydi",
                "complaint_text": "Bu kayit schema analizini calistirmak icin yeterli uzunluktadir ve gecerlidir.",
                "normalized_category": "OTHER",
                "tags": [],
                "support_count": 0,
                "is_synthetic": False,
                "quality_flags": [],
            }
        ),
        encoding="utf-8",
    )

    records: list[dict] = []
    for index in range(30):
        records.append(
            _normalized_record(
                record_id=f"g-{index:03d}",
                text=f"Mobil baglanti ve hizmet sorunu tekrarliyor. Kayit {index} icin detay metni bulunuyor.",
                created_at=f"2025-02-{(index % 27) + 1:02d}T11:00:00+00:00",
            )
        )

    artifacts_dir = tmp_path / "artifacts"
    splits_dir = artifacts_dir / "splits"
    labeled_path = artifacts_dir / "complaints_labeled.jsonl"
    _write_jsonl(labeled_path, records)
    _write_jsonl(splits_dir / "train.jsonl", records[:18])
    _write_jsonl(splits_dir / "val.jsonl", records[18:22])
    _write_jsonl(splits_dir / "test.jsonl", records[22:26])
    _write_jsonl(splits_dir / "hard_test.jsonl", records[26:30])

    step_rows = [
        _solution_step_record("STEP.OTHER.001", "L1", "Genel sorun tespiti", "GENERAL_SUPPORT"),
        _solution_step_record("STEP.OTHER.002", "L1", "Temel baglanti testi", "GENERAL_SUPPORT"),
        _solution_step_record("STEP.OTHER.003", "L2", "Destek kaydi hazirligi", "GENERAL_SUPPORT"),
    ]
    _write_jsonl(artifacts_dir / "solution_steps.jsonl", step_rows)

    kb_rows = [
        _kb_record("KB.OTHER.001", "KB.OTHER.001#P1", "STEP.OTHER.001", 0.82),
        _kb_record("KB.OTHER.002", "KB.OTHER.002#P1", "STEP.OTHER.002", 0.8),
        _kb_record("KB.OTHER.003", "KB.OTHER.003#P1", "STEP.OTHER.003", 0.78),
    ]
    _write_jsonl(artifacts_dir / "kb.jsonl", kb_rows)

    link_rows = [
        {
            "step_id": "STEP.OTHER.001",
            "evidence_ids": ["KB.OTHER.001#P1"],
            "rationale": "Temel tespit adimini destekler.",
            "version": "solution-steps-v1",
        },
        {
            "step_id": "STEP.OTHER.002",
            "evidence_ids": ["KB.OTHER.002#P1"],
            "rationale": "Baglanti kontrol adimini destekler.",
            "version": "solution-steps-v1",
        },
        {
            "step_id": "STEP.OTHER.003",
            "evidence_ids": ["KB.OTHER.003#P1"],
            "rationale": "Kayit hazirlik adimini destekler.",
            "version": "solution-steps-v1",
        },
    ]
    _write_jsonl(artifacts_dir / "step_kb_links.jsonl", link_rows)

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
paths:
  dataset: {raw_dataset}
  artifacts_dir: {artifacts_dir}
  schema_report: {artifacts_dir / "schema_report.json"}
  labeled_complaints: {labeled_path}
  train_split: {splits_dir / "train.jsonl"}
  val_split: {splits_dir / "val.jsonl"}
  test_split: {splits_dir / "test.jsonl"}
  hard_test_split: {splits_dir / "hard_test.jsonl"}
  solution_steps_jsonl: {artifacts_dir / "solution_steps.jsonl"}
  kb_jsonl: {artifacts_dir / "kb.jsonl"}
  step_kb_links_jsonl: {artifacts_dir / "step_kb_links.jsonl"}
  graph_nodes: {artifacts_dir / "graph" / "nodes.jsonl"}
  graph_edges: {artifacts_dir / "graph" / "edges.jsonl"}
  graph_stats: {artifacts_dir / "graph" / "graph_stats.json"}
  gnn_embeddings: {artifacts_dir / "graph" / "gnn_embeddings.npz"}
  embeddings_dir: {artifacts_dir / "embeddings"}
  retrieval_eval_json: {artifacts_dir / "retrieval_eval.json"}
  retrieval_eval_md: {artifacts_dir / "retrieval_eval.md"}
  review_pack_for_humans: {artifacts_dir / "review_pack_for_humans.jsonl"}
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
graph:
  use_gnn: {str(use_gnn).lower()}
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
    lambda_gnn: 0.05
    min_steps: 3
    top_steps: 5
    max_evidence: 10
    escalation_threshold: 0.58
  gnn:
    epochs: 3
    self_weight: 0.65
    neighbor_weight: 0.35
    convergence_tol: 0.0005
  evaluation:
    review_pack_size: 12
  mode:
    SMOKE:
      complaint_limit: null
      eval_limit: null
      include_retrieval_debug: false
    FULL:
      complaint_limit: null
      eval_limit: null
      include_retrieval_debug: true
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
    return config_path, artifacts_dir


def test_graph_stage_smoke_generates_required_artifacts(tmp_path: Path) -> None:
    config_path, artifacts_dir = _build_stage_graph_fixture(tmp_path=tmp_path, use_gnn=False)
    exit_code = run_pipeline(mode="SMOKE", config_path=str(config_path), stage="graph")
    assert exit_code == 0

    assert (artifacts_dir / "graph" / "nodes.jsonl").exists()
    assert (artifacts_dir / "graph" / "edges.jsonl").exists()
    assert (artifacts_dir / "graph" / "graph_stats.json").exists()
    assert (artifacts_dir / "retrieval_eval.json").exists()
    assert (artifacts_dir / "retrieval_eval.md").exists()
    assert (artifacts_dir / "review_pack_for_humans.jsonl").exists()

    retrieval_eval = json.loads((artifacts_dir / "retrieval_eval.json").read_text(encoding="utf-8"))
    assert retrieval_eval["weak_label_predictions"]["test_count"] > 0
    assert retrieval_eval["weak_label_predictions"]["hard_test_count"] > 0
    assert "evidence_quality_metrics" in retrieval_eval
    assert retrieval_eval["use_gnn"] is False


@pytest.mark.parametrize("use_gnn", [False, True])
def test_graph_stage_gnn_toggle(tmp_path: Path, use_gnn: bool) -> None:
    config_path, artifacts_dir = _build_stage_graph_fixture(tmp_path=tmp_path, use_gnn=use_gnn)
    exit_code = run_pipeline(mode="SMOKE", config_path=str(config_path), stage="graph")
    assert exit_code == 0

    retrieval_eval = json.loads((artifacts_dir / "retrieval_eval.json").read_text(encoding="utf-8"))
    gnn_path = artifacts_dir / "graph" / "gnn_embeddings.npz"
    if use_gnn:
        assert gnn_path.exists()
        assert retrieval_eval["use_gnn"] is True
        assert "gnn_embeddings_path" in retrieval_eval
    else:
        assert not gnn_path.exists()
        assert retrieval_eval["use_gnn"] is False
