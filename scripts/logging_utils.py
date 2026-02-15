from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        if hasattr(record, "event"):
            payload["event"] = getattr(record, "event")
        if hasattr(record, "payload"):
            payload["payload"] = getattr(record, "payload")
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def configure_json_logging(level: str, log_file: str | None = None) -> logging.Logger:
    logger = logging.getLogger("telecom_pipeline")
    logger.setLevel(level.upper())
    logger.handlers.clear()
    logger.propagate = False

    formatter = JsonFormatter()

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_file:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def log_event(logger: logging.Logger, level: str, event: str, payload: dict[str, Any]) -> None:
    logger.log(
        getattr(logging, level.upper(), logging.INFO),
        event,
        extra={"event": event, "payload": payload},
    )

