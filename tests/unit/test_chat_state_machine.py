from __future__ import annotations

from api.chat_service import ChatService


class _FakeEngine:
    mode = "SMOKE"

    def infer(self, complaint_text: str, brand: str | None = None):
        return {
            "request_id": "INFER-TEST",
            "generation_mode": "renderer_fallback",
            "response_text": (
                "1) Tanı: OTHER (Güven: 0.80)\n"
                "2) Netleştirme Soruları (gerekliyse): Gerekmiyor.\n"
                "3) Çözüm Adımları (sırayla, 3–5 adım):\n"
                "- [STEP:STEP.OTHER.001] Test adimi. (Kanıt: KB.OTHER.001#P1)\n"
                "- [STEP:STEP.OTHER.002] Test adimi. (Kanıt: KB.OTHER.002#P1)\n"
                "- [STEP:STEP.OTHER.003] Test adimi. (Kanıt: KB.OTHER.003#P1)\n"
                "4) Beklenen Sonuç / Kontrol:\n"
                "Kontrol\n"
                "5) Çözülmediyse Eskalasyon:\n"
                "- Birim: GENERAL_SUPPORT | Neden: test\n"
                "6) Soru: “Sorununuz çözüldü mü? (Evet/Hayır)”"
            ),
            "validation": {
                "template_compliant": True,
                "step_valid": True,
                "evidence_valid": True,
                "pii_free": True,
                "final_question_present": True,
                "is_valid": True,
                "extracted_step_ids": ["STEP.OTHER.001"],
                "extracted_evidence_ids": ["KB.OTHER.001#P1"],
                "evidence_coverage": 1.0,
                "missing_sections": [],
                "violations": [],
            },
            "evidence_pack": {
                "normalized_category": "OTHER",
                "category_confidence": 0.8,
                "top_steps": [
                    {
                        "step_id": "STEP.OTHER.001",
                        "title_tr": "Adim1",
                        "level": "L1",
                        "instructions_tr": ["a", "b", "c"],
                        "evidence_ids": ["KB.OTHER.001#P1"],
                    },
                    {
                        "step_id": "STEP.OTHER.002",
                        "title_tr": "Adim2",
                        "level": "L1",
                        "instructions_tr": ["a", "b", "c"],
                        "evidence_ids": ["KB.OTHER.002#P1"],
                    },
                    {
                        "step_id": "STEP.OTHER.003",
                        "title_tr": "Adim3",
                        "level": "L2",
                        "instructions_tr": ["a", "b", "c"],
                        "evidence_ids": ["KB.OTHER.003#P1"],
                    },
                ],
                "evidence": [
                    {"paragraph_id": "KB.OTHER.001#P1", "text_tr": "e1", "confidence": 0.8},
                    {"paragraph_id": "KB.OTHER.002#P1", "text_tr": "e2", "confidence": 0.8},
                    {"paragraph_id": "KB.OTHER.003#P1", "text_tr": "e3", "confidence": 0.8},
                ],
                "escalation_suggestion": {
                    "unit": "GENERAL_SUPPORT",
                    "reason": "test reason",
                    "threshold_signals": [],
                },
            },
            "latency_ms": 2.0,
            "model_backend_reason": "test",
            "safety_assessment": {
                "should_refuse": False,
                "is_security_attack": False,
                "is_data_exfiltration": False,
                "matched_rules": [],
            },
        }


def test_chat_state_machine_escalates_after_max_attempts() -> None:
    service = ChatService(engine=_FakeEngine(), max_attempts=2)
    start = service.start_session("internet sorunu")
    session_id = start["session_id"]
    assert start["stage"] == "AWAITING_RESOLUTION"

    first_no = service.continue_session(session_id, "Hayır")
    assert first_no["stage"] == "AWAITING_CLARIFICATION"

    second_try = service.continue_session(session_id, "Aksam saatlerinde artiyor")
    assert second_try["stage"] == "AWAITING_RESOLUTION"

    escalated = service.continue_session(session_id, "Hayır")
    assert escalated["status"] == "ESCALATED"
    assert escalated["agent_handoff_packet"] is not None
    assert escalated["agent_handoff_packet"]["schema_name"] == "LiveAgentHandoffPacket"


def test_chat_state_machine_resolve_and_feedback_close() -> None:
    service = ChatService(engine=_FakeEngine(), max_attempts=2)
    start = service.start_session("ses sorunu")
    session_id = start["session_id"]

    resolved = service.continue_session(session_id, "Evet")
    assert resolved["stage"] == "AWAITING_FEEDBACK"

    closed = service.continue_session(session_id, "5")
    assert closed["status"] == "CLOSED"
