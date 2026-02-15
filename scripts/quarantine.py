from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def append_quarantine_record(
    quarantine_path: str | Path,
    line_number: int,
    raw_record: Any,
    reason: str,
    details: Any,
) -> None:
    path = Path(quarantine_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "line_number": line_number,
        "reason": reason,
        "details": details,
        "raw_record": raw_record,
    }
    with path.open("a", encoding="utf-8") as outfile:
        outfile.write(json.dumps(payload, ensure_ascii=False))
        outfile.write("\n")

