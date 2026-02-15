from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from data.schemas import NormalizedComplaint


def load_normalized_jsonl(path: str | Path) -> list[NormalizedComplaint]:
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"Required split file not found: {target}")
    rows: list[NormalizedComplaint] = []
    with target.open("r", encoding="utf-8") as infile:
        for line in infile:
            line = line.strip()
            if not line:
                continue
            rows.append(NormalizedComplaint.model_validate(json.loads(line)))
    return rows


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_markdown(path: str | Path, lines: list[str]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines), encoding="utf-8")


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.array(values, dtype=np.float64), q))


def run_inference_records(
    *,
    engine: Any,
    records: list[NormalizedComplaint],
    split_name: str,
    mode: str,
) -> tuple[list[dict[str, Any]], float]:
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    for record in records:
        try:
            result = engine.infer(record.complaint_text_clean, brand=record.brand_slug)
            error_message = None
        except Exception as error:
            if mode == "FULL":
                raise RuntimeError(
                    f"Inference failed in FULL mode for split={split_name} complaint_id={record.complaint_id}: {error}"
                ) from error
            error_message = str(error)
            result = {
                "request_id": f"INFER-ERR-{record.complaint_id}",
                "generation_mode": "smoke_inference_error",
                "response_text": "Geçici bir hata oluştu. Güvenli destek akışı için canlı temsilciye aktarım önerilir.",
                "validation": {
                    "template_compliant": False,
                    "step_valid": True,
                    "evidence_valid": True,
                    "pii_free": True,
                    "final_question_present": False,
                    "is_valid": False,
                    "extracted_step_ids": [],
                    "extracted_evidence_ids": [],
                    "evidence_coverage": 0.0,
                    "missing_sections": ["INFERENCE_ERROR"],
                    "violations": ["INFERENCE_ERROR"],
                },
                "evidence_pack": {
                    "normalized_category": "OTHER",
                    "category_confidence": 0.0,
                    "top_steps": [],
                    "evidence": [],
                    "escalation_suggestion": {
                        "unit": "GENERAL_SUPPORT",
                        "reason": "Inceleme icin canlı destek aktarımı önerilir.",
                        "threshold_signals": ["INFERENCE_ERROR"],
                    },
                },
                "latency_ms": 0.0,
                "model_backend_reason": error_message,
                "safety_assessment": {
                    "should_refuse": False,
                    "is_security_attack": False,
                    "is_data_exfiltration": False,
                    "matched_rules": [],
                },
            }
        rows.append(
            {
                "split": split_name,
                "complaint_id": record.complaint_id,
                "complaint_text_clean": record.complaint_text_clean,
                "true_category": record.normalized_category,
                "source_category": record.source_category,
                "quality_flags": record.quality_flags,
                "inference": result,
                "inference_error": error_message,
            }
        )
    elapsed = time.perf_counter() - started
    return rows, elapsed
