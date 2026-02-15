from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from data.schemas import EvaluationReport, RawComplaint


def test_raw_complaint_accepts_valid_record() -> None:
    record = RawComplaint.model_validate(
        {
            "complaint_id": "123",
            "url": "https://www.example.com/test",
            "brand_name": "DemoTel",
            "brand_slug": "demotel",
            "title": "Hizmet kesintisi",
            "complaint_text": "Internet hizmetim son 2 gundur sik sik kesiliyor ve cozulmuyor.",
            "created_at_iso": "2026-01-01T10:00:00",
            "scraped_at_iso": "2026-01-02T12:00:00+00:00",
            "normalized_category": "INTERNET",
            "original_category_label": "Kesinti",
            "tags": ["internet", "kesinti"],
            "support_count": 3,
            "is_synthetic": False,
            "quality_flags": [],
            "http_status": 200,
            "parse_version": "v1",
            "source_complaint_id": "456",
        }
    )
    assert record.schema_name == "RawComplaint"


def test_raw_complaint_rejects_extra_field() -> None:
    with pytest.raises(ValidationError):
        RawComplaint.model_validate(
            {
                "brand_name": "DemoTel",
                "brand_slug": "demotel",
                "title": "x",
                "complaint_text": "Bu metin test icin yeterince uzundur ve en az 20 karakterdir.",
                "normalized_category": "INTERNET",
                "tags": [],
                "support_count": 0,
                "is_synthetic": False,
                "quality_flags": [],
                "unexpected": "field",
            }
        )


def test_evaluation_report_count_guard() -> None:
    with pytest.raises(ValidationError):
        EvaluationReport.model_validate(
            {
                "run_id": "run-1",
                "mode": "SMOKE",
                "dataset_size": 10,
                "valid_records": 8,
                "quarantined_records": 5,
                "hallucination_rate": 0.0,
                "evidence_coverage": 1.0,
                "escalation_rate": 0.0,
                "latency_p95_ms": 0.0,
                "pass_fail": "PASS",
                "notes": [],
                "metrics": {},
                "generated_at_iso": datetime.now(timezone.utc).isoformat(),
            }
        )

