from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from api.app import create_app


class _FakeChatService:
    def start_session(self, complaint_text: str):
        return {
            "session_id": "S-1",
            "status": "OPEN",
            "stage": "AWAITING_RESOLUTION",
            "attempts": 1,
            "assistant_message": "Yanıt",
            "prompt_for_user": "Sorununuz çözüldü mü? (Evet/Hayır)",
            "steps": [],
            "evidence": [],
            "citations": [],
            "escalation_decision": None,
            "agent_handoff_packet": None,
        }

    def continue_session(self, session_id: str, user_response: str):
        return {
            "session_id": session_id,
            "status": "CLOSED",
            "stage": "CLOSED",
            "attempts": 1,
            "assistant_message": "Kapatıldı",
            "prompt_for_user": None,
            "steps": [],
            "evidence": [],
            "citations": [],
            "escalation_decision": None,
            "agent_handoff_packet": None,
        }

    def get_state(self, session_id: str):
        return {"session_id": session_id, "status": "OPEN", "stage": "AWAITING_RESOLUTION", "attempts": 1}

    def get_evidence_pack(self, session_id: str):
        return {"request_id": "R1", "normalized_category": "OTHER", "top_steps": [], "evidence": []}


def test_api_endpoints_return_expected_schema() -> None:
    app = create_app(chat_service=_FakeChatService(), config_path="config.yaml", mode="SMOKE")
    client = TestClient(app)

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"

    start = client.post("/chat/start", json={"complaint_text": "internet baglanti sorunu var"})
    assert start.status_code == 200
    payload = start.json()
    assert "assistant_message" in payload
    assert "steps" in payload
    assert "evidence" in payload
    assert "citations" in payload

    cont = client.post("/chat/continue", json={"session_id": "S-1", "user_response": "Evet"})
    assert cont.status_code == 200
    assert cont.json()["status"] == "CLOSED"

    state = client.get("/chat/state", params={"session_id": "S-1"})
    assert state.status_code == 200
    assert state.json()["session_id"] == "S-1"

    pack = client.get("/evidence_pack", params={"session_id": "S-1"})
    assert pack.status_code == 200
    assert "normalized_category" in pack.json()
