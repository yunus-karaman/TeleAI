from __future__ import annotations

import hashlib
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from data.schemas import EvidencePack, KBParagraph, NormalizedComplaint, SolutionStep
from graph.embeddings import HashingTextEmbedder
from graph.retrieval import build_retrieval_resources, retrieve_evidence_pack
from solution_steps.generator import StepKBLink
from taxonomy.schema import load_taxonomy_file


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _normalized(complaint_id: str, text: str, category: str = "OTHER") -> NormalizedComplaint:
    return NormalizedComplaint(
        complaint_id=complaint_id,
        brand_name="DemoTel",
        brand_slug="demotel",
        created_at_iso="2025-01-01T10:00:00+00:00",
        title_clean="test",
        complaint_text_clean=text,
        normalized_category=category,
        confidence_score=0.8,
        assignment_reason="test assignment",
        needs_review=False,
        source_category=category,
        quality_flags=[],
        duplicate_cluster_id=None,
        is_duplicate_of=None,
        taxonomy_version="1.0.0",
        source_hash_sha256=_sha(complaint_id + text),
    )


def _solution_step(step_id: str, title: str, level: str) -> SolutionStep:
    return SolutionStep(
        step_id=step_id,
        category_id="OTHER",
        level=level,
        title_tr=title,
        instructions_tr=[
            "Sorunun hangi zamanda ciktigini not edin.",
            "Ayni denemeyi farkli bir agda tekrar edin.",
            "Gorulen hata mesajini not ederek kaydedin.",
        ],
        required_inputs=["device_os", "error_code", "city_region"],
        success_check="Sorun tekrari ve kapsam bilgisi netlesir.",
        stop_conditions=["Iki denemede de duzelmezse destek birimine aktar."],
        escalation_unit="GENERAL_SUPPORT",
        risk_level="low" if level == "L1" else "medium",
        tags=["other", "test"],
        version="solution-steps-v1",
    )


def _kb_paragraph(doc_id: str, paragraph_id: str, step_id: str, text: str, confidence: float) -> KBParagraph:
    return KBParagraph(
        doc_id=doc_id,
        paragraph_id=paragraph_id,
        text_tr=text,
        applies_to_step_ids=[step_id],
        source_type="internal_best_practice",
        confidence=confidence,
        version="solution-steps-v1",
    )


def _build_resources():
    taxonomy = load_taxonomy_file("taxonomy/taxonomy.yaml")
    complaints = [
        _normalized("c-001", "Mobil internet cok yavas ve baglanti surekli kopuyor, gun icinde tekrar ediyor."),
        _normalized("c-002", "Hizmette genel sorun yasiyorum, uygulama baglanti hatasi veriyor ve tekrarliyor."),
        _normalized("c-003", "Destek adimlari sonrasinda sorun devam ediyor, ek teknik inceleme gerekiyor."),
    ]

    steps = sorted(
        [
            _solution_step("STEP.OTHER.001", "Genel sorun tespiti", "L1"),
            _solution_step("STEP.OTHER.002", "Temel baglanti testi", "L1"),
            _solution_step("STEP.OTHER.003", "Destek kaydi hazirligi", "L2"),
        ],
        key=lambda item: item.step_id,
    )

    kb_items = sorted(
        [
            _kb_paragraph(
                "KB.OTHER.001",
                "KB.OTHER.001#P1",
                "STEP.OTHER.001",
                "Temel sorun tespiti, tekrar kosullarini netlestirerek yanlis yonlendirmeyi azaltir.",
                0.82,
            ),
            _kb_paragraph(
                "KB.OTHER.002",
                "KB.OTHER.002#P1",
                "STEP.OTHER.002",
                "Baglanti testi farkli ortamda yapildiginda problemin yerel mi genel mi oldugu ayrisir.",
                0.8,
            ),
            _kb_paragraph(
                "KB.OTHER.003",
                "KB.OTHER.003#P1",
                "STEP.OTHER.003",
                "Destek kaydi oncesi teknik ozet hazirlamak inceleme suresini kisaltir ve tutarlilik saglar.",
                0.78,
            ),
        ],
        key=lambda item: item.paragraph_id,
    )

    links = sorted(
        [
            StepKBLink(
                step_id="STEP.OTHER.001",
                evidence_ids=["KB.OTHER.001#P1"],
                rationale="Temel tespit adimini destekler.",
                version="solution-steps-v1",
            ),
            StepKBLink(
                step_id="STEP.OTHER.002",
                evidence_ids=["KB.OTHER.002#P1"],
                rationale="Baglanti test adimini destekler.",
                version="solution-steps-v1",
            ),
            StepKBLink(
                step_id="STEP.OTHER.003",
                evidence_ids=["KB.OTHER.003#P1"],
                rationale="Kayit hazirlik adimini destekler.",
                version="solution-steps-v1",
            ),
        ],
        key=lambda item: item.step_id,
    )

    embedder = HashingTextEmbedder(dimension=96, ngram_min=1, ngram_max=2)

    complaint_ids = [item.complaint_id for item in complaints]
    complaint_texts = [item.complaint_text_clean for item in complaints]

    step_ids = [item.step_id for item in steps]
    step_texts = [f"{item.title_tr} {' '.join(item.instructions_tr)}" for item in steps]

    kb_ids = [item.paragraph_id for item in kb_items]
    kb_texts = [item.text_tr for item in kb_items]

    category_rows = sorted(taxonomy.categories, key=lambda item: item.category_id)
    category_ids = [item.category_id for item in category_rows]
    category_texts = [f"{item.title_tr} {item.description_tr} {' '.join(item.keywords_tr)}" for item in category_rows]

    embeddings = {
        "complaints": (complaint_ids, embedder.embed(complaint_texts)),
        "steps": (step_ids, embedder.embed(step_texts)),
        "kb": (kb_ids, embedder.embed(kb_texts)),
        "categories": (category_ids, embedder.embed(category_texts)),
    }

    resources = build_retrieval_resources(
        taxonomy=taxonomy,
        complaints=complaints,
        steps=steps,
        kb_items=kb_items,
        links=links,
        embeddings=embeddings,
        embedder_callable=embedder.embed,
        config={
            "taxonomy_assignment": {
                "min_confidence": 0.55,
                "low_confidence_policy": "other",
                "review_margin_threshold": 0.08,
                "rule_weight": 0.55,
                "embedding_weight": 0.45,
                "keyword_weight": 1.0,
                "negative_weight": 0.8,
                "example_weight": 1.2,
                "embedding": {
                    "max_features": 3000,
                    "ngram_min": 1,
                    "ngram_max": 2,
                    "min_df": 1,
                },
            },
            "seed": 42,
            "alpha": 0.45,
            "beta": 0.15,
            "gamma": 0.25,
            "delta": 0.10,
            "lambda_gnn": 0.0,
            "top_steps": 5,
            "min_steps": 3,
            "max_evidence": 10,
            "escalation_threshold": 0.58,
        },
        step_prior_weight={step.step_id: 0.75 for step in steps},
        step_gnn_embeddings=None,
        category_gnn_embeddings=None,
    )
    return resources


def test_evidence_pack_validation_rejects_unknown_evidence_reference() -> None:
    with pytest.raises(ValidationError):
        EvidencePack(
            request_id="REQ-1",
            normalized_category="OTHER",
            category_confidence=0.7,
            top_steps=[
                EvidencePack.TopStepItem(
                    step_id="STEP.OTHER.001",
                    title_tr="Test",
                    level="L1",
                    instructions_tr=["birinci adim aciklamasi", "ikinci adim aciklamasi", "ucuncu adim aciklamasi"],
                    evidence_ids=["KB.OTHER.001#P9"],
                    step_score=0.7,
                )
            ],
            evidence=[
                EvidencePack.EvidenceItem(
                    paragraph_id="KB.OTHER.001#P1",
                    text_tr="Bu paragraf destekleyici bir aciklamadir.",
                    confidence=0.7,
                )
            ],
            escalation_suggestion=EvidencePack.EscalationSuggestion(
                unit="GENERAL_SUPPORT",
                reason="Test nedeni",
                threshold_signals=[],
            ),
            retrieval_debug=None,
        )


def test_graph_retrieval_is_deterministic_for_same_input() -> None:
    resources = _build_resources()
    complaint_text = "Mobil baglanti yavas ve surekli kopuyor, sorun tekrarliyor."

    pack1, telemetry1 = retrieve_evidence_pack(
        complaint_text=complaint_text,
        resources=resources,
        request_id="REQ-A",
        include_debug=False,
    )
    pack2, telemetry2 = retrieve_evidence_pack(
        complaint_text=complaint_text,
        resources=resources,
        request_id="REQ-B",
        include_debug=False,
    )

    assert pack1.normalized_category == pack2.normalized_category
    assert [item.step_id for item in pack1.top_steps] == [item.step_id for item in pack2.top_steps]
    assert [item.step_score for item in pack1.top_steps] == [item.step_score for item in pack2.top_steps]
    assert [item.paragraph_id for item in pack1.evidence] == [item.paragraph_id for item in pack2.evidence]
    assert telemetry1["step_count"] == telemetry2["step_count"]
    assert telemetry1["evidence_count"] == telemetry2["evidence_count"]


def test_graph_retrieval_never_returns_step_without_evidence() -> None:
    resources = _build_resources()
    pack, _ = retrieve_evidence_pack(
        complaint_text="Yavaslayan internet ve tekrar eden baglanti kopmasi icin yardim istiyorum.",
        resources=resources,
        request_id="REQ-EVIDENCE",
        include_debug=False,
    )

    evidence_ids = {item.paragraph_id for item in pack.evidence}
    assert 3 <= len(pack.top_steps) <= 5
    for step in pack.top_steps:
        assert len(step.evidence_ids) >= 1
        for evidence_id in step.evidence_ids:
            assert evidence_id in evidence_ids


def test_graph_retrieval_falls_back_to_category_with_steps_when_assignment_has_none() -> None:
    resources = _build_resources()

    class _NoStepCategoryAssigner:
        def assign(self, text: str) -> SimpleNamespace:  # noqa: ARG002
            return SimpleNamespace(
                normalized_category="BILLING_PAYMENTS",
                confidence_score=0.31,
                assignment_reason="forced-for-test",
                combined_scores={"BILLING_PAYMENTS": 0.99, "OTHER": 0.01},
            )

    resources.taxonomy_assigner = _NoStepCategoryAssigner()
    pack, _ = retrieve_evidence_pack(
        complaint_text="Baglanti sorunu icin destek gerekiyor.",
        resources=resources,
        request_id="REQ-FALLBACK",
        include_debug=True,
    )

    assert pack.normalized_category == "OTHER"
    assert len(pack.top_steps) >= 1
    assert pack.retrieval_debug is not None
    assignment = pack.retrieval_debug["category_assignment"]
    assert assignment["requested_category"] == "BILLING_PAYMENTS"
    assert assignment["fallback_category_used"] is True
