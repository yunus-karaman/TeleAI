from __future__ import annotations

import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TR_STOPWORDS = {
    "ve",
    "bir",
    "bu",
    "da",
    "de",
    "ile",
    "icin",
    "için",
    "cok",
    "çok",
    "daha",
    "gibi",
    "ama",
    "ben",
    "bana",
    "mi",
    "mı",
    "neden",
    "nasil",
    "nasıl",
    "hizmet",
    "hizmeti",
    "musteri",
    "müşteri",
    "sorun",
    "paket",
    "internet",
    "hat",
    "sikayet",
    "şikayet",
}
EN_STOPWORDS = {
    "the",
    "and",
    "is",
    "are",
    "to",
    "for",
    "with",
    "that",
    "this",
    "it",
    "of",
    "in",
    "on",
    "as",
    "be",
    "or",
}
TR_SPECIAL_CHARS = set("çğıöşüÇĞİÖŞÜ")


def _infer_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    if isinstance(value, list):
        return "list"
    if isinstance(value, dict):
        return "dict"
    return type(value).__name__


def _estimate_language(text: str) -> str:
    if not text:
        return "unknown"

    letters = [char for char in text if char.isalpha()]
    if len(letters) < 20:
        return "unknown"

    lowered = text.lower()
    tokens: list[str] = []
    current: list[str] = []
    for char in lowered:
        if char.isalpha() or char == "'":
            current.append(char)
        elif current:
            tokens.append("".join(current))
            current = []
    if current:
        tokens.append("".join(current))

    token_count = max(len(tokens), 1)
    tr_hits = sum(1 for token in tokens if token in TR_STOPWORDS)
    en_hits = sum(1 for token in tokens if token in EN_STOPWORDS)
    tr_char_ratio = sum(1 for char in text if char in TR_SPECIAL_CHARS) / max(len(text), 1)
    tr_score = (tr_hits / token_count) + tr_char_ratio
    en_score = en_hits / token_count

    if tr_score >= 0.04 and tr_score > en_score * 1.2:
        return "tr"
    if en_score >= 0.05 and en_score > tr_score * 1.2:
        return "en"
    return "mixed_or_other"


def analyze_dataset_schema(dataset_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    dataset_path = Path(dataset_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    field_presence: Counter[str] = Counter()
    field_null: Counter[str] = Counter()
    field_type_counts: defaultdict[str, Counter[str]] = defaultdict(Counter)
    language_counts: Counter[str] = Counter()
    all_fields: set[str] = set()

    text_lengths: list[int] = []
    parse_errors = 0
    malformed_lines: list[int] = []
    empty_text_count = 0
    non_string_text_count = 0
    total_records = 0

    with dataset_path.open("r", encoding="utf-8") as infile:
        for line_number, raw_line in enumerate(infile, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                parse_errors += 1
                if len(malformed_lines) < 20:
                    malformed_lines.append(line_number)
                continue

            if not isinstance(record, dict):
                parse_errors += 1
                if len(malformed_lines) < 20:
                    malformed_lines.append(line_number)
                continue

            total_records += 1
            fields = set(record.keys())
            all_fields.update(fields)

            for field in fields:
                value = record[field]
                field_presence[field] += 1
                field_type_counts[field][_infer_type(value)] += 1
                if value is None:
                    field_null[field] += 1

            complaint_text = record.get("complaint_text")
            if complaint_text is None:
                field_null["complaint_text"] += 1
            elif isinstance(complaint_text, str):
                stripped = complaint_text.strip()
                if stripped:
                    text_lengths.append(len(stripped))
                    language_counts[_estimate_language(stripped)] += 1
                else:
                    empty_text_count += 1
            else:
                non_string_text_count += 1

    if total_records == 0:
        raise RuntimeError("No valid records parsed from dataset.")

    field_statistics: dict[str, Any] = {}
    null_percentages: dict[str, float] = {}
    inconsistencies: list[str] = []

    for field in sorted(all_fields):
        present_count = field_presence[field]
        missing_count = total_records - present_count
        null_count = field_null[field]
        missing_or_null_count = missing_count + null_count
        present_rate = present_count / total_records
        missing_rate = missing_count / total_records
        missing_or_null_rate = missing_or_null_count / total_records
        null_rate_when_present = null_count / present_count if present_count else 1.0

        types = [
            {
                "type": key,
                "count": count,
                "rate": round(count / present_count, 6) if present_count else 0.0,
            }
            for key, count in field_type_counts[field].most_common()
        ]

        required = present_count == total_records and null_count == 0
        field_statistics[field] = {
            "required": required,
            "present_count": present_count,
            "present_rate": round(present_rate, 6),
            "missing_count": missing_count,
            "missing_rate": round(missing_rate, 6),
            "null_count": null_count,
            "null_rate_when_present": round(null_rate_when_present, 6),
            "missing_or_null_count": missing_or_null_count,
            "missing_or_null_rate": round(missing_or_null_rate, 6),
            "inferred_types": types,
        }
        null_percentages[field] = round(100.0 * missing_or_null_rate, 4)

        if len(field_type_counts[field]) > 1:
            inconsistencies.append(f"Field '{field}' has multiple types: {dict(field_type_counts[field])}")
        if missing_count > 0:
            inconsistencies.append(f"Field '{field}' missing in {missing_count} records ({missing_rate:.2%})")
        if null_count > 0:
            inconsistencies.append(
                f"Field '{field}' null in {null_count} present records ({null_rate_when_present:.2%})"
            )

    if parse_errors > 0:
        inconsistencies.append(
            f"JSON parse/object errors on {parse_errors} lines. Sample line numbers: {malformed_lines}"
        )
    if empty_text_count > 0:
        inconsistencies.append(f"Empty complaint_text in {empty_text_count} records")
    if non_string_text_count > 0:
        inconsistencies.append(f"Non-string complaint_text in {non_string_text_count} records")

    sorted_lengths = sorted(text_lengths)
    if sorted_lengths:
        p95_index = max(0, math.ceil(0.95 * len(sorted_lengths)) - 1)
        p95_length = int(sorted_lengths[p95_index])
        distribution = {
            "count": len(sorted_lengths),
            "min": int(sorted_lengths[0]),
            "max": int(sorted_lengths[-1]),
            "median": float(statistics.median(sorted_lengths)),
            "p90": int(sorted_lengths[max(0, math.ceil(0.90 * len(sorted_lengths)) - 1)]),
            "p95": p95_length,
            "p99": int(sorted_lengths[max(0, math.ceil(0.99 * len(sorted_lengths)) - 1)]),
            "avg": round(sum(sorted_lengths) / len(sorted_lengths), 3),
        }
        avg_length = distribution["avg"]
    else:
        p95_length = 0
        distribution = {
            "count": 0,
            "min": 0,
            "max": 0,
            "median": 0,
            "p90": 0,
            "p95": 0,
            "p99": 0,
            "avg": 0,
        }
        avg_length = 0

    language_total = sum(language_counts.values())
    language_distribution = {
        language: {"count": count, "rate": round(count / language_total, 6)}
        for language, count in language_counts.most_common()
    }

    report = {
        "dataset_path": str(dataset_path),
        "total_records": total_records,
        "total_fields": len(all_fields),
        "field_statistics": field_statistics,
        "null_percentages": null_percentages,
        "avg_text_length": avg_length,
        "95th_percentile_length": p95_length,
        "text_length_distribution": distribution,
        "language_distribution_estimate": language_distribution,
        "required_fields": [name for name, info in field_statistics.items() if info["required"]],
        "optional_fields": [name for name, info in field_statistics.items() if not info["required"]],
        "inconsistencies": inconsistencies,
        "quality_notes": {
            "parse_errors": parse_errors,
            "empty_complaint_text_records": empty_text_count,
            "non_string_complaint_text_records": non_string_text_count,
        },
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report

