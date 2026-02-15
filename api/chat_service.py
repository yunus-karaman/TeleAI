from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from data.schemas import LiveAgentHandoffPacket
from models.infer import ConstrainedInferenceEngine
from preprocess.pii import sanitize_for_artifact


def _is_yes(text: str) -> bool:
    lowered = (text or "").strip().lower()
    return lowered in {"evet", "yes", "cozuldu", "çözüldü", "ok"}


def _is_no(text: str) -> bool:
    lowered = (text or "").strip().lower()
    return lowered in {"hayır", "hayir", "no", "cozulmedi", "çözülmedi"}


def _default_clarifying_questions() -> list[str]:
    return [
        "Sorun en cok hangi saat araliginda tekrar ediyor?",
        "Ayni sorunu farkli cihaz veya ag ile denediniz mi?",
    ]


@dataclass
class ChatSession:
    session_id: str
    complaint_text: str
    mode: str
    max_attempts: int
    attempts: int = 0
    status: str = "OPEN"
    stage: str = "AWAITING_RESOLUTION"
    clarifying_answers: list[str] = field(default_factory=list)
    last_inference: dict[str, Any] | None = None
    handoff_packet: dict[str, Any] | None = None
    feedback: int | None = None


class ChatService:
    def __init__(self, *, engine: ConstrainedInferenceEngine, max_attempts: int = 2) -> None:
        self.engine = engine
        self.max_attempts = max_attempts
        self.sessions: dict[str, ChatSession] = {}

    def _build_response_payload(
        self,
        *,
        session: ChatSession,
        assistant_message: str,
        prompt_for_user: str | None = None,
    ) -> dict[str, Any]:
        inference = session.last_inference or {}
        pack = inference.get("evidence_pack", {})
        steps = [
            {
                "step_id": item["step_id"],
                "title_tr": item["title_tr"],
                "level": item["level"],
                "instructions_tr": item["instructions_tr"],
                "evidence_ids": item["evidence_ids"],
            }
            for item in pack.get("top_steps", [])
        ]
        evidence = [
            {
                "paragraph_id": item["paragraph_id"],
                "text_tr": item["text_tr"],
                "confidence": item["confidence"],
            }
            for item in pack.get("evidence", [])
        ]
        citations = sorted({evidence_id for step in steps for evidence_id in step["evidence_ids"]})

        return {
            "session_id": session.session_id,
            "status": session.status,
            "stage": session.stage,
            "attempts": session.attempts,
            "assistant_message": assistant_message,
            "prompt_for_user": prompt_for_user,
            "steps": steps,
            "evidence": evidence,
            "citations": citations,
            "escalation_decision": pack.get("escalation_suggestion"),
            "agent_handoff_packet": session.handoff_packet,
        }

    def _run_inference(self, session: ChatSession, complaint_text: str) -> dict[str, Any]:
        result = self.engine.infer(complaint_text)
        session.last_inference = result
        session.attempts += 1
        return result

    def _make_handoff_packet(self, session: ChatSession, reason: str) -> dict[str, Any]:
        if session.last_inference is None:
            raise RuntimeError("Cannot create handoff packet without inference result.")
        pack = session.last_inference["evidence_pack"]
        step_ids = [item["step_id"] for item in pack.get("top_steps", [])]
        evidence_ids = sorted({evidence_id for item in pack.get("top_steps", []) for evidence_id in item["evidence_ids"]})
        packet = LiveAgentHandoffPacket(
            session_id=session.session_id,
            normalized_category=pack["normalized_category"],
            category_confidence=float(pack["category_confidence"]),
            complaint_summary_masked=sanitize_for_artifact(session.complaint_text, max_chars=800),
            steps_tried=step_ids,
            evidence_ids_used=evidence_ids,
            clarifying_answers=[sanitize_for_artifact(ans, max_chars=200) for ans in session.clarifying_answers],
            recommended_escalation_unit=pack["escalation_suggestion"]["unit"],
            escalation_reason=reason,
            attempts=session.attempts,
            generated_at_iso=datetime.now(timezone.utc).isoformat(),
        )
        return packet.model_dump(mode="json")

    def start_session(self, complaint_text: str) -> dict[str, Any]:
        session_id = f"S-{uuid.uuid4().hex[:12]}"
        session = ChatSession(
            session_id=session_id,
            complaint_text=complaint_text,
            mode=self.engine.mode,
            max_attempts=self.max_attempts,
        )
        self.sessions[session_id] = session
        inference = self._run_inference(session, complaint_text)
        message = inference["response_text"]
        session.stage = "AWAITING_RESOLUTION"
        return self._build_response_payload(
            session=session,
            assistant_message=message,
            prompt_for_user="Sorununuz çözüldü mü? (Evet/Hayır)",
        )

    def continue_session(self, session_id: str, user_response: str) -> dict[str, Any]:
        if session_id not in self.sessions:
            raise KeyError(f"Unknown session_id: {session_id}")
        session = self.sessions[session_id]
        text = (user_response or "").strip()

        if session.stage == "AWAITING_RESOLUTION":
            if _is_yes(text):
                session.stage = "AWAITING_FEEDBACK"
                session.status = "RESOLVED"
                return self._build_response_payload(
                    session=session,
                    assistant_message="Harika. Oturumu kapatmadan once yaniti degerlendirebilir misiniz?",
                    prompt_for_user="Bu yanıt faydalı mı? (1–5)",
                )
            if _is_no(text):
                if session.attempts < session.max_attempts:
                    session.stage = "AWAITING_CLARIFICATION"
                    questions = _default_clarifying_questions()
                    return self._build_response_payload(
                        session=session,
                        assistant_message="Anladım. Yeniden denemek için kısa netleştirme soruları soruyorum:",
                        prompt_for_user="\n".join([f"- {question}" for question in questions]),
                    )
                session.stage = "ESCALATED"
                session.status = "ESCALATED"
                session.handoff_packet = self._make_handoff_packet(
                    session,
                    reason="Maksimum deneme sayisina ulasildi ve sorun devam ediyor.",
                )
                return self._build_response_payload(
                    session=session,
                    assistant_message="Sizi canlı müşteri temsilcisine aktarıyorum.",
                    prompt_for_user=None,
                )
            return self._build_response_payload(
                session=session,
                assistant_message="Lütfen yalnızca 'Evet' veya 'Hayır' yanıtı verin.",
                prompt_for_user="Sorununuz çözüldü mü? (Evet/Hayır)",
            )

        if session.stage == "AWAITING_CLARIFICATION":
            session.clarifying_answers.append(text)
            refined = f"{session.complaint_text}\nEk bilgi: {text}"
            inference = self._run_inference(session, refined)
            session.stage = "AWAITING_RESOLUTION"
            return self._build_response_payload(
                session=session,
                assistant_message=inference["response_text"],
                prompt_for_user="Sorununuz çözüldü mü? (Evet/Hayır)",
            )

        if session.stage == "AWAITING_FEEDBACK":
            try:
                score = int(text)
            except ValueError:
                score = 0
            if not (1 <= score <= 5):
                return self._build_response_payload(
                    session=session,
                    assistant_message="Lütfen 1 ile 5 arasında bir puan verin.",
                    prompt_for_user="Bu yanıt faydalı mı? (1–5)",
                )
            session.feedback = score
            session.stage = "CLOSED"
            session.status = "CLOSED"
            return self._build_response_payload(
                session=session,
                assistant_message="Teşekkür ederim. Oturum kapatıldı.",
                prompt_for_user=None,
            )

        return self._build_response_payload(
            session=session,
            assistant_message="Bu oturum kapatılmıştır veya canlı temsilciye aktarılmıştır.",
            prompt_for_user=None,
        )

    def get_state(self, session_id: str) -> dict[str, Any]:
        if session_id not in self.sessions:
            raise KeyError(f"Unknown session_id: {session_id}")
        session = self.sessions[session_id]
        return {
            "session_id": session.session_id,
            "status": session.status,
            "stage": session.stage,
            "attempts": session.attempts,
            "max_attempts": session.max_attempts,
            "clarifying_answers": session.clarifying_answers,
            "feedback": session.feedback,
            "has_handoff_packet": session.handoff_packet is not None,
        }

    def get_evidence_pack(self, session_id: str) -> dict[str, Any]:
        if session_id not in self.sessions:
            raise KeyError(f"Unknown session_id: {session_id}")
        session = self.sessions[session_id]
        if session.last_inference is None:
            raise RuntimeError("Session has no inference output yet.")
        return session.last_inference.get("evidence_pack", {})
