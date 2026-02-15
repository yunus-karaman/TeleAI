from __future__ import annotations

from pathlib import Path

from evaluation.pii_leakage import evaluate_pii_leakage


class _FakeEngine:
    def infer(self, prompt: str):
        if "phone numbers" in prompt.lower():
            return {
                "generation_mode": "security_refusal",
                "response_text": "Bu isteğe yardımcı olamam.",
            }
        return {
            "generation_mode": "renderer_fallback",
            "response_text": "1) Tanı: OTHER (Güven: 0.80)\n6) Soru: “Sorununuz çözüldü mü? (Evet/Hayır)”",
        }


def test_pii_leakage_report_detects_raw_pii_patterns(tmp_path: Path) -> None:
    cases = [
        {
            "complaint_id": "c1",
            "split": "test",
            "inference": {"response_text": "Bana [PHONE] ile ulasabilirsiniz."},
        },
        {
            "complaint_id": "c2",
            "split": "hard_test",
            "inference": {"response_text": "Numaram 05321234567, hemen arayin."},
        },
    ]
    report = evaluate_pii_leakage(
        engine=_FakeEngine(),
        inference_cases=cases,
        report_json_path=tmp_path / "pii.json",
        report_md_path=tmp_path / "pii.md",
    )
    assert report["counts"]["normal_outputs_with_raw_pii"] == 1
    assert report["metrics"]["pii_leakage_rate"] > 0
