from __future__ import annotations

import hashlib

from data.schemas import NormalizedComplaint
from taxonomy.splitting import create_splits


def _hash_for_id(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def test_duplicate_clusters_do_not_cross_splits() -> None:
    records: list[NormalizedComplaint] = []
    for index in range(12):
        cluster_id = "dup_cluster_1" if index in (0, 1, 2) else "dup_cluster_2" if index in (3, 4) else None
        records.append(
            NormalizedComplaint(
                complaint_id=f"c-{index:03d}",
                brand_name="DemoTel",
                brand_slug="demotel",
                created_at_iso=f"2025-01-{index+1:02d}T10:00:00",
                title_clean="Test",
                complaint_text_clean=f"Test complaint text number {index} with enough content for schema validation.",
                normalized_category="BILLING_PAYMENTS" if index % 2 == 0 else "ROAMING_INTERNATIONAL",
                confidence_score=0.8,
                assignment_reason="test reason",
                needs_review=False,
                source_category="FATURA_ODEME",
                quality_flags=[],
                duplicate_cluster_id=cluster_id,
                is_duplicate_of=None,
                taxonomy_version="1.0.0",
                source_hash_sha256=_hash_for_id(str(index)),
            )
        )

    split = create_splits(
        records=records,
        split_config={
            "train_ratio": 0.7,
            "val_ratio": 0.1,
            "test_ratio": 0.2,
            "hard_test_ratio": 0.1,
            "hard_test_max_size": 5,
            "hard_short_max_chars": 100,
            "hard_confidence_threshold": 0.65,
        },
    )

    cluster_to_splits: dict[str, set[str]] = {}
    for split_name, split_records in {
        "train": split.train,
        "val": split.val,
        "test": split.test,
    }.items():
        for record in split_records:
            if not record.duplicate_cluster_id:
                continue
            cluster_to_splits.setdefault(record.duplicate_cluster_id, set()).add(split_name)

    assert all(len(split_names) == 1 for split_names in cluster_to_splits.values())

