from __future__ import annotations

import pytest
from pydantic import ValidationError

from data.schemas import LiveAgentHandoffPacket


def test_live_agent_handoff_packet_valid() -> None:
    packet = LiveAgentHandoffPacket(
        session_id="S-123",
        normalized_category="OTHER",
        category_confidence=0.75,
        complaint_summary_masked="Kullanici baglanti sorununun devam ettigini belirtiyor.",
        steps_tried=["STEP.OTHER.001"],
        evidence_ids_used=["KB.OTHER.001#P1"],
        clarifying_answers=["Aksam saatlerinde tekrar ediyor."],
        recommended_escalation_unit="GENERAL_SUPPORT",
        escalation_reason="Maksimum deneme asildi.",
        attempts=2,
        generated_at_iso="2026-02-15T10:00:00+00:00",
    )
    assert packet.schema_name == "LiveAgentHandoffPacket"


def test_live_agent_handoff_packet_rejects_bad_attempts() -> None:
    with pytest.raises(ValidationError):
        LiveAgentHandoffPacket(
            session_id="S-123",
            normalized_category="OTHER",
            category_confidence=0.75,
            complaint_summary_masked="Kullanici baglanti sorununun devam ettigini belirtiyor.",
            steps_tried=["STEP.OTHER.001"],
            evidence_ids_used=["KB.OTHER.001#P1"],
            clarifying_answers=["Aksam saatlerinde tekrar ediyor."],
            recommended_escalation_unit="GENERAL_SUPPORT",
            escalation_reason="Maksimum deneme asildi.",
            attempts=0,
            generated_at_iso="2026-02-15T10:00:00+00:00",
        )
