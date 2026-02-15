from __future__ import annotations

import json
from pathlib import Path

from data.ingestion import load_and_validate_raw_complaints


def test_ingestion_quarantines_invalid_records(tmp_path: Path) -> None:
    dataset = tmp_path / "health.jsonl"
    dataset.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "brand_name": "DemoTel",
                        "brand_slug": "demotel",
                        "title": "Hizmet sorunu",
                        "complaint_text": "Bu metin yeterince uzun ve validasyon asamasindan gecmesi gerekir.",
                        "normalized_category": "INTERNET",
                        "tags": [],
                        "support_count": 0,
                        "is_synthetic": False,
                        "quality_flags": [],
                    }
                ),
                json.dumps({"title": "Eksik alanli kayit"}),
            ]
        ),
        encoding="utf-8",
    )

    quarantine = tmp_path / "quarantine.jsonl"
    valid_records, stats = load_and_validate_raw_complaints(dataset, quarantine, sample_size=None)
    assert len(valid_records) == 1
    assert stats["quarantined_records"] == 1
    assert quarantine.exists()

