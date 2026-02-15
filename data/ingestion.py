from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from data.schemas import RawComplaint
from scripts.quarantine import append_quarantine_record


def load_and_validate_raw_complaints(
    dataset_path: str | Path,
    quarantine_path: str | Path,
    sample_size: int | None = None,
) -> tuple[list[RawComplaint], dict[str, int]]:
    dataset_path = Path(dataset_path)
    quarantine_path = Path(quarantine_path)
    quarantine_path.parent.mkdir(parents=True, exist_ok=True)
    quarantine_path.write_text("", encoding="utf-8")

    valid: list[RawComplaint] = []
    stats = {
        "total_lines": 0,
        "valid_records": 0,
        "quarantined_records": 0,
        "json_parse_errors": 0,
        "schema_validation_errors": 0,
        "non_object_records": 0,
    }

    with dataset_path.open("r", encoding="utf-8") as infile:
        for line_number, raw_line in enumerate(infile, start=1):
            if sample_size is not None and stats["valid_records"] >= sample_size:
                break

            stats["total_lines"] += 1
            line = raw_line.strip()
            if not line:
                append_quarantine_record(
                    quarantine_path=quarantine_path,
                    line_number=line_number,
                    raw_record=line,
                    reason="empty_line",
                    details="Line was empty after trimming.",
                )
                stats["quarantined_records"] += 1
                continue

            try:
                loaded: Any = json.loads(line)
            except json.JSONDecodeError as error:
                append_quarantine_record(
                    quarantine_path=quarantine_path,
                    line_number=line_number,
                    raw_record=line,
                    reason="json_parse_error",
                    details=str(error),
                )
                stats["quarantined_records"] += 1
                stats["json_parse_errors"] += 1
                continue

            if not isinstance(loaded, dict):
                append_quarantine_record(
                    quarantine_path=quarantine_path,
                    line_number=line_number,
                    raw_record=loaded,
                    reason="non_object_record",
                    details="JSON value is not an object.",
                )
                stats["quarantined_records"] += 1
                stats["non_object_records"] += 1
                continue

            try:
                complaint = RawComplaint.model_validate(loaded)
            except ValidationError as error:
                append_quarantine_record(
                    quarantine_path=quarantine_path,
                    line_number=line_number,
                    raw_record=loaded,
                    reason="schema_validation_error",
                    details=error.errors(include_url=False),
                )
                stats["quarantined_records"] += 1
                stats["schema_validation_errors"] += 1
                continue

            valid.append(complaint)
            stats["valid_records"] += 1

    return valid, stats

