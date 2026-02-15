from __future__ import annotations

from pathlib import Path

from evaluation.hallucination import evaluate_hallucination


def test_hallucination_scoring_detects_invalid_step_and_missing_evidence(tmp_path: Path) -> None:
    cases = [
        {
            "split": "test",
            "complaint_id": "c-valid",
            "inference": {
                "response_text": (
                    "1) Tanı: OTHER (Güven: 0.80)\n"
                    "2) Netleştirme Soruları (gerekliyse): Gerekmiyor.\n"
                    "3) Çözüm Adımları (sırayla, 3–5 adım):\n"
                    "- [STEP:STEP.OTHER.001] Test adimi. (Kanıt: KB.OTHER.001#P1)\n"
                    "4) Beklenen Sonuç / Kontrol:\n"
                    "Tamam\n"
                    "5) Çözülmediyse Eskalasyon:\n"
                    "- Birim: GENERAL_SUPPORT | Neden: test\n"
                    "6) Soru: “Sorununuz çözüldü mü? (Evet/Hayır)”"
                ),
                "validation": {"template_compliant": True},
                "evidence_pack": {
                    "top_steps": [{"step_id": "STEP.OTHER.001"}],
                    "evidence": [{"paragraph_id": "KB.OTHER.001#P1"}],
                },
            },
        },
        {
            "split": "test",
            "complaint_id": "c-invalid",
            "inference": {
                "response_text": (
                    "1) Tanı: OTHER (Güven: 0.70)\n"
                    "2) Netleştirme Soruları (gerekliyse): Gerekmiyor.\n"
                    "3) Çözüm Adımları (sırayla, 3–5 adım):\n"
                    "- [STEP:STEP.OTHER.999] Uydurma adim.\n"
                    "4) Beklenen Sonuç / Kontrol:\n"
                    "Tamam\n"
                    "5) Çözülmediyse Eskalasyon:\n"
                    "- Birim: GENERAL_SUPPORT | Neden: test\n"
                    "6) Soru: “Sorununuz çözüldü mü? (Evet/Hayır)”"
                ),
                "validation": {"template_compliant": False},
                "evidence_pack": {
                    "top_steps": [{"step_id": "STEP.OTHER.001"}],
                    "evidence": [{"paragraph_id": "KB.OTHER.001#P1"}],
                },
            },
        },
    ]

    report = evaluate_hallucination(
        inference_cases=cases,
        report_json_path=tmp_path / "hallucination.json",
        report_md_path=tmp_path / "hallucination.md",
    )
    assert report["metrics"]["hallucination_rate_actionable"] > 0
    assert report["metrics"]["step_hallucination_rate"] > 0
    assert report["metrics"]["template_compliance_rate"] < 1.0
