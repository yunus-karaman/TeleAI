from __future__ import annotations

import hashlib
import json

from data.schemas import EvidencePack, NormalizedComplaint
from training.data_builder import build_task2_examples


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _record(complaint_id: str, text: str) -> NormalizedComplaint:
    return NormalizedComplaint(
        complaint_id=complaint_id,
        brand_name="DemoTel",
        brand_slug="demotel",
        created_at_iso="2025-01-01T10:00:00+00:00",
        title_clean="test",
        complaint_text_clean=text,
        normalized_category="OTHER",
        confidence_score=0.8,
        assignment_reason="test assignment",
        needs_review=False,
        source_category="OTHER",
        quality_flags=[],
        duplicate_cluster_id=None,
        is_duplicate_of=None,
        taxonomy_version="1.0.0",
        source_hash_sha256=_sha(complaint_id + text),
    )


def _pack() -> EvidencePack:
    return EvidencePack(
        request_id="REQ-DETERMINISTIC",
        normalized_category="OTHER",
        category_confidence=0.72,
        top_steps=[
            EvidencePack.TopStepItem(
                step_id="STEP.OTHER.001",
                title_tr="Tespit",
                level="L1",
                instructions_tr=[
                    "Belirtiyi tek cumle ile netlestirin.",
                    "Sorunun tekrar saatini kaydedin.",
                    "Ayni testi farkli agda uygulayin.",
                ],
                evidence_ids=["KB.OTHER.001#P1"],
                step_score=0.9,
            ),
            EvidencePack.TopStepItem(
                step_id="STEP.OTHER.002",
                title_tr="Dogrulama",
                level="L1",
                instructions_tr=[
                    "Baglantiyi yeniden baslatin.",
                    "Kisa sure sonra tekrar deneyin.",
                    "Hata mesaji varsa not edin.",
                ],
                evidence_ids=["KB.OTHER.002#P1"],
                step_score=0.85,
            ),
            EvidencePack.TopStepItem(
                step_id="STEP.OTHER.003",
                title_tr="Eskalasyon hazirligi",
                level="L2",
                instructions_tr=[
                    "Test sonuclarini tek bir listede toplayin.",
                    "Sorunun etkisini olculebilir yazin.",
                    "Destek kaydi icin ozet metin hazirlayin.",
                ],
                evidence_ids=["KB.OTHER.003#P1"],
                step_score=0.8,
            ),
        ],
        evidence=[
            EvidencePack.EvidenceItem(
                paragraph_id="KB.OTHER.001#P1",
                text_tr="Temel tespit adimi sorunu siniflandirir.",
                confidence=0.8,
            ),
            EvidencePack.EvidenceItem(
                paragraph_id="KB.OTHER.002#P1",
                text_tr="Dogrulama adimi yerel/genel ayrimi destekler.",
                confidence=0.78,
            ),
            EvidencePack.EvidenceItem(
                paragraph_id="KB.OTHER.003#P1",
                text_tr="Eskalasyon oncesi kayit ozetini guclendirir.",
                confidence=0.76,
            ),
        ],
        escalation_suggestion=EvidencePack.EscalationSuggestion(
            unit="GENERAL_SUPPORT",
            reason="Sorun surerse resmi destek kaydi acilmali.",
            threshold_signals=[],
        ),
        retrieval_debug=None,
    )


def test_task2_dataset_construction_is_deterministic() -> None:
    records = [
        _record("c-001", "Mobil internet baglantisi kopuyor ve yavasliyor."),
        _record("c-002", "Hat cekmesine ragmen hizmette tekrarli kesinti yasaniyor."),
    ]

    def provider(record: NormalizedComplaint, _resources, _debug):
        pack = _pack()
        pack.request_id = f"REQ-{record.complaint_id}"
        return pack, {}

    first = build_task2_examples(
        records=records,
        split="train",
        resources=object(),  # provider does not use retrieval resources in this test
        dataset_version="task2-sft-v1",
        min_steps=3,
        max_steps=5,
        include_debug=False,
        retrieval_pack_provider=provider,
    )
    second = build_task2_examples(
        records=records,
        split="train",
        resources=object(),
        dataset_version="task2-sft-v1",
        min_steps=3,
        max_steps=5,
        include_debug=False,
        retrieval_pack_provider=provider,
    )

    first_dump = [json.dumps(item.model_dump(mode="json"), ensure_ascii=False, sort_keys=True) for item in first]
    second_dump = [json.dumps(item.model_dump(mode="json"), ensure_ascii=False, sort_keys=True) for item in second]
    assert first_dump == second_dump
