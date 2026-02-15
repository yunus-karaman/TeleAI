from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _artifacts_dir(config: dict[str, Any]) -> Path:
    return Path(config.get("paths", {}).get("artifacts_dir", "artifacts"))


def aborted_reason_path(config: dict[str, Any]) -> Path:
    configured = config.get("paths", {}).get("aborted_reason")
    if configured:
        return Path(str(configured))
    return _artifacts_dir(config) / "aborted_reason.json"


def smoke_notice_path(config: dict[str, Any]) -> Path:
    configured = config.get("paths", {}).get("smoke_notice")
    if configured:
        return Path(str(configured))
    return _artifacts_dir(config) / "smoke_notice.json"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_aborted_reason(
    config: dict[str, Any],
    *,
    stage: str,
    reason_code: str,
    message: str,
    details: dict[str, Any] | None = None,
) -> Path:
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "stage": stage,
        "reason_code": reason_code,
        "message": message,
        "details": details or {},
    }
    target = aborted_reason_path(config)
    _write_json(target, payload)
    return target


def append_smoke_notice(
    config: dict[str, Any],
    *,
    stage: str,
    notice_code: str,
    message: str,
    details: dict[str, Any] | None = None,
) -> Path:
    target = smoke_notice_path(config)
    notices: list[dict[str, Any]]
    if target.exists():
        try:
            existing = json.loads(target.read_text(encoding="utf-8"))
            notices = existing.get("notices", []) if isinstance(existing, dict) else []
        except json.JSONDecodeError:
            notices = []
    else:
        notices = []

    notices.append(
        {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "stage": stage,
            "notice_code": notice_code,
            "message": message,
            "details": details or {},
        }
    )
    payload = {"count": len(notices), "notices": notices}
    _write_json(target, payload)
    return target


def fail_fast_enabled(config: dict[str, Any], gate_key: str) -> bool:
    fail_cfg = config.get("fail_fast", {})
    if not isinstance(fail_cfg, dict):
        return False
    return bool(fail_cfg.get(gate_key, False))


def handle_gate_violation(
    *,
    config: dict[str, Any],
    mode: str,
    stage: str,
    gate_key: str,
    reason_code: str,
    message: str,
    details: dict[str, Any] | None = None,
    logger: Any | None = None,
) -> None:
    if mode == "FULL" and fail_fast_enabled(config, gate_key):
        abort_path = write_aborted_reason(
            config,
            stage=stage,
            reason_code=reason_code,
            message=message,
            details=details,
        )
        if logger is not None:
            logger.error(
                "fail_fast_abort",
                extra={
                    "event": "fail_fast_abort",
                    "payload": {
                        "stage": stage,
                        "gate_key": gate_key,
                        "reason_code": reason_code,
                        "message": message,
                        "aborted_reason_path": str(abort_path),
                    },
                },
            )
        raise RuntimeError(f"[{reason_code}] {message} (aborted_reason={abort_path})")

    notice_path = append_smoke_notice(
        config,
        stage=stage,
        notice_code=reason_code,
        message=message,
        details=details,
    )
    if logger is not None:
        logger.warning(
            "smoke_notice",
            extra={
                "event": "smoke_notice",
                "payload": {
                    "stage": stage,
                    "gate_key": gate_key,
                    "notice_code": reason_code,
                    "message": message,
                    "smoke_notice_path": str(notice_path),
                },
            },
        )
