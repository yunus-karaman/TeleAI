from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from data.schemas import NormalizedComplaint


SLANG_KEYWORDS = {
    "kanka",
    "abi",
    "ya",
    "yaa",
    "napcam",
    "napicam",
    "rezalet",
    "berbat",
    "sacma",
    "valla",
}
EDGE_CASE_CATEGORIES = {
    "ROAMING_INTERNATIONAL",
    "NUMBER_PORTING_MNP",
    "CONTRACT_COMMITMENT_CANCELLATION",
}


@dataclass(frozen=True)
class SplitResult:
    train: list[NormalizedComplaint]
    val: list[NormalizedComplaint]
    test: list[NormalizedComplaint]
    hard_test: list[NormalizedComplaint]
    split_assignments: dict[str, str]


def _parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _stable_hash(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:12], 16)


def _is_slang_or_typo(text: str) -> bool:
    lowered = text.lower()
    if any(keyword in lowered for keyword in SLANG_KEYWORDS):
        return True
    repeated_chars = re.findall(r"(.)\1{3,}", lowered)
    punctuation_noise = len(re.findall(r"[!?]{3,}", lowered))
    return bool(repeated_chars) or punctuation_noise > 0


def _is_multi_issue(record: NormalizedComplaint) -> bool:
    if "MULTI_COMPLAINT_SUSPECTED" in record.quality_flags:
        return True
    lowered = record.complaint_text_clean.lower()
    connectors = ["ayrica", "bir de", "hem ", "aynı zamanda", "diger yandan"]
    hits = sum(1 for connector in connectors if connector in lowered)
    return hits >= 2


def _make_groups(records: list[NormalizedComplaint]) -> dict[str, list[NormalizedComplaint]]:
    grouped: dict[str, list[NormalizedComplaint]] = defaultdict(list)
    for record in records:
        key = record.duplicate_cluster_id or f"single::{record.complaint_id}"
        grouped[key].append(record)
    return grouped


def _assign_groups_to_splits(
    groups: dict[str, list[NormalizedComplaint]],
    train_ratio: float,
    val_ratio: float,
) -> dict[str, str]:
    total_records = sum(len(items) for items in groups.values())
    target_train = int(total_records * train_ratio)
    target_val = int(total_records * val_ratio)

    dated_groups: list[tuple[str, datetime, int]] = []
    undated_groups: list[tuple[str, int]] = []
    for group_id, items in groups.items():
        dates = [dt for dt in (_parse_datetime(item.created_at_iso) for item in items) if dt is not None]
        if dates:
            dated_groups.append((group_id, min(dates), len(items)))
        else:
            undated_groups.append((group_id, len(items)))

    dated_groups.sort(key=lambda item: (item[1], item[0]))
    undated_groups.sort(key=lambda item: (_stable_hash(item[0]), item[0]))

    split_assignments: dict[str, str] = {}
    counts = {"train": 0, "val": 0, "test": 0}

    for group_id, _dt, group_size in dated_groups:
        if counts["train"] < target_train:
            split = "train"
        elif counts["val"] < target_val:
            split = "val"
        else:
            split = "test"
        split_assignments[group_id] = split
        counts[split] += group_size

    for group_id, group_size in undated_groups:
        deficits = {
            "train": target_train - counts["train"],
            "val": target_val - counts["val"],
            "test": (total_records - target_train - target_val) - counts["test"],
        }
        split = sorted(deficits.items(), key=lambda item: (-item[1], item[0]))[0][0]
        split_assignments[group_id] = split
        counts[split] += group_size

    return split_assignments


def _select_hard_test(
    test_records: list[NormalizedComplaint],
    all_records_count: int,
    config: dict[str, Any],
) -> list[NormalizedComplaint]:
    if not test_records:
        return []

    ratio = float(config["hard_test_ratio"])
    max_size = int(config["hard_test_max_size"])
    target_size = min(max_size, max(1, int(all_records_count * ratio)))
    target_size = min(target_size, len(test_records))

    short_max_chars = int(config["hard_short_max_chars"])
    confidence_threshold = float(config["hard_confidence_threshold"])
    per_criterion_limit = max(1, target_size // 5)

    sorted_test = sorted(test_records, key=lambda item: (item.confidence_score, item.complaint_id))
    by_id = {record.complaint_id: record for record in sorted_test}

    candidates_low_conf = [record.complaint_id for record in sorted_test if record.confidence_score <= confidence_threshold]
    candidates_edge = [record.complaint_id for record in sorted_test if record.normalized_category in EDGE_CASE_CATEGORIES]
    candidates_short = [
        record.complaint_id for record in sorted_test if len(record.complaint_text_clean) <= short_max_chars
    ]
    candidates_slang = [record.complaint_id for record in sorted_test if _is_slang_or_typo(record.complaint_text_clean)]
    candidates_multi = [record.complaint_id for record in sorted_test if _is_multi_issue(record)]

    selected_ids: list[str] = []
    seen = set()

    def add_candidates(candidate_ids: list[str], limit: int) -> None:
        for complaint_id in candidate_ids:
            if complaint_id in seen:
                continue
            selected_ids.append(complaint_id)
            seen.add(complaint_id)
            if len([identifier for identifier in selected_ids if identifier in set(candidate_ids)]) >= limit:
                break
            if len(selected_ids) >= target_size:
                break

    for pool in [candidates_low_conf, candidates_edge, candidates_short, candidates_slang, candidates_multi]:
        add_candidates(pool, limit=per_criterion_limit)
        if len(selected_ids) >= target_size:
            break

    if len(selected_ids) < target_size:
        for record in sorted_test:
            if record.complaint_id not in seen:
                selected_ids.append(record.complaint_id)
                seen.add(record.complaint_id)
            if len(selected_ids) >= target_size:
                break

    return [by_id[identifier] for identifier in selected_ids]


def create_splits(records: list[NormalizedComplaint], split_config: dict[str, Any]) -> SplitResult:
    groups = _make_groups(records)
    assignments = _assign_groups_to_splits(
        groups=groups,
        train_ratio=float(split_config["train_ratio"]),
        val_ratio=float(split_config["val_ratio"]),
    )

    train: list[NormalizedComplaint] = []
    val: list[NormalizedComplaint] = []
    test: list[NormalizedComplaint] = []
    for group_id, items in groups.items():
        split = assignments[group_id]
        target = train if split == "train" else val if split == "val" else test
        target.extend(items)

    train = sorted(train, key=lambda item: item.complaint_id)
    val = sorted(val, key=lambda item: item.complaint_id)
    test = sorted(test, key=lambda item: item.complaint_id)
    hard_test = _select_hard_test(
        test_records=test,
        all_records_count=len(records),
        config=split_config,
    )

    return SplitResult(
        train=train,
        val=val,
        test=test,
        hard_test=hard_test,
        split_assignments=assignments,
    )

