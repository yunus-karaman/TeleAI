from __future__ import annotations

from data.schemas import EvidencePack
from models.template_renderer import render_deterministic_response
from models.validation import validate_response_against_pack


def _pack() -> EvidencePack:
    return EvidencePack(
        request_id="REQ-VAL",
        normalized_category="OTHER",
        category_confidence=0.74,
        top_steps=[
            EvidencePack.TopStepItem(
                step_id="STEP.OTHER.001",
                title_tr="Tespit",
                level="L1",
                instructions_tr=[
                    "Belirtiyi tek cumle ile aciklayin.",
                    "Sorunun tekrar zamanini kaydedin.",
                    "Farkli ag ile tekrar deneyin.",
                ],
                evidence_ids=["KB.OTHER.001#P1"],
                step_score=0.92,
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
                step_score=0.87,
            ),
            EvidencePack.TopStepItem(
                step_id="STEP.OTHER.003",
                title_tr="Eskalasyon",
                level="L2",
                instructions_tr=[
                    "Test bulgularini tek listede toplayin.",
                    "Sorunun etkisini olculebilir yazin.",
                    "Destek icin ozet kaydi hazirlayin.",
                ],
                evidence_ids=["KB.OTHER.003#P1"],
                step_score=0.83,
            ),
        ],
        evidence=[
            EvidencePack.EvidenceItem(
                paragraph_id="KB.OTHER.001#P1",
                text_tr="Tespit adimi sorun siniflandirmasini destekler.",
                confidence=0.8,
            ),
            EvidencePack.EvidenceItem(
                paragraph_id="KB.OTHER.002#P1",
                text_tr="Dogrulama adimi tekrar kosullarini netlestirir.",
                confidence=0.78,
            ),
            EvidencePack.EvidenceItem(
                paragraph_id="KB.OTHER.003#P1",
                text_tr="Eskalasyon adimi resmi takip surecini kolaylastirir.",
                confidence=0.76,
            ),
        ],
        escalation_suggestion=EvidencePack.EscalationSuggestion(
            unit="GENERAL_SUPPORT",
            reason="Sorun surerse resmi destek birimine aktarim gerekir.",
            threshold_signals=[],
        ),
        retrieval_debug=None,
    )


def test_template_validator_accepts_deterministic_renderer_output() -> None:
    pack = _pack()
    response = render_deterministic_response(pack, min_steps=3, max_steps=5)
    validation = validate_response_against_pack(response, pack)
    assert validation.template_compliant
    assert validation.step_valid
    assert validation.evidence_valid
    assert validation.pii_free
    assert validation.is_valid


def test_template_validator_rejects_step_and_evidence_outside_pack() -> None:
    pack = _pack()
    bad_response = (
        "1) Tanı: OTHER (Güven: 0.70)\n"
        "2) Netleştirme Soruları (gerekliyse): Gerekmiyor.\n"
        "3) Çözüm Adımları (sırayla, 3–5 adım):\n"
        "- [STEP:STEP.OTHER.999] Ornek adim. (Kanıt: KB.OTHER.999#P1)\n"
        "4) Beklenen Sonuç / Kontrol:\n"
        "Kontrol metni.\n"
        "5) Çözülmediyse Eskalasyon:\n"
        "- Birim: GENERAL_SUPPORT | Neden: test\n"
        "6) Soru: “Sorununuz çözüldü mü? (Evet/Hayır)”"
    )
    validation = validate_response_against_pack(bad_response, pack)
    assert not validation.step_valid
    assert not validation.evidence_valid
    assert not validation.is_valid
