from __future__ import annotations

import json
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, ConfigDict, Field

from api.chat_service import ChatService
from models.infer import ConstrainedInferenceEngine
from scripts.config_loader import load_config
from scripts.logging_utils import configure_json_logging
from scripts.reproducibility import set_global_determinism


class StartRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    complaint_text: str = Field(min_length=8)


class ContinueRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    session_id: str = Field(min_length=1)
    user_response: str = Field(min_length=1)


def _build_ui_html() -> str:
    return """
<!doctype html>
<html lang="tr">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Telekom Asistan Demo</title>
  <style>
    body { font-family: "Segoe UI", Tahoma, sans-serif; margin: 0; background: #f4f7fb; color: #0f172a; }
    .wrap { display: grid; grid-template-columns: 1.4fr 1fr; gap: 16px; padding: 16px; }
    .card { background: #fff; border: 1px solid #dbe3ef; border-radius: 10px; padding: 12px; }
    .title { font-weight: 700; margin-bottom: 8px; }
    .chat-log { height: 420px; overflow: auto; white-space: pre-wrap; border: 1px solid #e5ecf5; border-radius: 8px; padding: 8px; background: #f8fbff; }
    textarea,input,button { width: 100%; box-sizing: border-box; padding: 10px; margin-top: 8px; border: 1px solid #cfd8e3; border-radius: 8px; }
    button { background: #0b5cab; color: white; border: none; font-weight: 600; cursor: pointer; }
    button:hover { background: #084c8f; }
    .pane { max-height: 180px; overflow: auto; white-space: pre-wrap; font-size: 13px; background: #fafcff; border: 1px solid #e5ecf5; border-radius: 8px; padding: 8px; }
    @media (max-width: 960px) { .wrap { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <div class="title">Sohbet</div>
      <div id="chatLog" class="chat-log"></div>
      <textarea id="complaint" rows="4" placeholder="Şikayetinizi yazın..."></textarea>
      <button onclick="startChat()">Sohbeti Başlat</button>
      <input id="userReply" placeholder="Yanıtınız (Evet/Hayır veya açıklama)" />
      <button onclick="continueChat()">Devam Et</button>
    </div>
    <div class="card">
      <div class="title">Adımlar</div>
      <div id="stepsPane" class="pane"></div>
      <div class="title">Kanıt</div>
      <div id="evidencePane" class="pane"></div>
      <div class="title">Agent Brief</div>
      <div id="handoffPane" class="pane"></div>
    </div>
  </div>
<script>
let sessionId = null;
function renderPayload(data) {
  const log = document.getElementById("chatLog");
  log.textContent += "\\nAsistan: " + (data.assistant_message || "") + "\\n";
  if (data.prompt_for_user) log.textContent += "Sistem: " + data.prompt_for_user + "\\n";
  document.getElementById("stepsPane").textContent = JSON.stringify(data.steps || [], null, 2);
  document.getElementById("evidencePane").textContent = JSON.stringify(data.evidence || [], null, 2);
  document.getElementById("handoffPane").textContent = JSON.stringify(data.agent_handoff_packet || {}, null, 2);
  log.scrollTop = log.scrollHeight;
}
async function startChat() {
  const complaint = document.getElementById("complaint").value;
  const res = await fetch("/chat/start", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ complaint_text: complaint })
  });
  const data = await res.json();
  sessionId = data.session_id;
  const log = document.getElementById("chatLog");
  log.textContent = "Kullanıcı: " + complaint + "\\n";
  renderPayload(data);
}
async function continueChat() {
  if (!sessionId) return;
  const reply = document.getElementById("userReply").value;
  const res = await fetch("/chat/continue", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, user_response: reply })
  });
  const data = await res.json();
  const log = document.getElementById("chatLog");
  log.textContent += "\\nKullanıcı: " + reply + "\\n";
  renderPayload(data);
}
</script>
</body>
</html>
""".strip()


def create_app(
    *,
    config_path: str = "config.yaml",
    mode: str = "SMOKE",
    chat_service: ChatService | None = None,
) -> FastAPI:
    config = load_config(config_path, mode)
    logger = configure_json_logging(level=config["logging"]["level"], log_file=config["logging"]["file"])
    set_global_determinism(seed=int(config["reproducibility"]["seed"]), deterministic=True)
    if chat_service is None:
        engine = ConstrainedInferenceEngine(config=config, mode=mode, logger=logger)
        max_attempts = int(config.get("evaluation", {}).get("chat", {}).get("max_attempts", 2))
        service = ChatService(engine=engine, max_attempts=max_attempts)
    else:
        service = chat_service

    app = FastAPI(title="Telecom Complaint Assistant Demo API", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "mode": mode}

    @app.get("/", response_class=HTMLResponse)
    def home() -> str:
        return _build_ui_html()

    @app.post("/chat/start")
    def chat_start(payload: StartRequest) -> dict[str, Any]:
        try:
            return service.start_session(payload.complaint_text)
        except Exception as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.post("/chat/continue")
    def chat_continue(payload: ContinueRequest) -> dict[str, Any]:
        try:
            return service.continue_session(payload.session_id, payload.user_response)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except Exception as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.get("/chat/state")
    def chat_state(session_id: str = Query(..., min_length=1)) -> dict[str, Any]:
        try:
            return service.get_state(session_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @app.get("/evidence_pack")
    def evidence_pack(session_id: str = Query(..., min_length=1)) -> dict[str, Any]:
        try:
            return service.get_evidence_pack(session_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except Exception as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    return app


app = create_app()
