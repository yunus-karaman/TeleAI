# api

## Responsibility
Service layer for request handling, policy enforcement, chat-state transitions, and live-agent escalation endpoints.

## Run
- `pip install fastapi uvicorn`
- `uvicorn api.app:app --host 0.0.0.0 --port 8000`

## Endpoints
- `POST /chat/start`
- `POST /chat/continue`
- `GET /chat/state?session_id=...`
- `GET /evidence_pack?session_id=...`
- `GET /health`
- `GET /` (minimal demo UI)

## Interface
Contract definitions live in interface.py.
