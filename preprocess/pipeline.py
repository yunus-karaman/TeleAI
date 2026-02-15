from __future__ import annotations

import json
import hashlib
import logging
import math
import statistics
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from data.schemas import CONTRACT_VERSION, CleanComplaint, RawComplaint
from preprocess.duplicates import DuplicateCluster, cluster_near_duplicates
from preprocess.pii import detect_pii_tags, mask_pii_text, sanitize_for_artifact
from preprocess.text_cleaning import (
    DEFAULT_SCRIPT_INDICATORS,
    assess_multi_complaint,
    assess_script_noise,
    clean_text_content,
    extract_primary_complaint,
)
from scripts.logging_utils import log_event


REASON_JSON_PARSE_ERROR = "JSON_PARSE_ERROR"
REASON_NON_OBJECT = "NON_OBJECT_JSON"
REASON_RAW_SCHEMA_INVALID = "RAW_SCHEMA_INVALID"
REASON_SCRIPT_NOISE = "SCRIPT_NOISE"
REASON_MULTI_COMPLAINT = "MULTI_COMPLAINT_CONTAMINATION"
REASON_PII_LEAK_AFTER_MASK = "PII_LEAK_AFTER_MASK"
REASON_TOO_SHORT = "TOO_SHORT"
REASON_CLEAN_SCHEMA_INVALID = "CLEAN_SCHEMA_INVALID"
REASON_INTERNAL_ERROR = "INTERNAL_PREPROCESS_ERROR"


@dataclass
class LoaderStats:
    total: int
    valid: int
    quarantined_by_reason: Counter[str]


def _append_quarantine_record(
    quarantine_path: Path,
    reason_code: str,
    error_message: str,
    schema_version: str,
    original_line: str | None = None,
    complaint_id: str | None = None,
    line_number: int | None = None,
) -> None:
    sanitized_error = sanitize_for_artifact(error_message, max_chars=500)
    if detect_pii_tags(sanitized_error, ignore_mask_tokens=True):
        sanitized_error = "[REDACTED_ERROR_MESSAGE]"

    payload: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "reason_code": reason_code,
        "error_message": sanitized_error,
        "schema_version": schema_version,
    }
    if original_line:
        sanitized_line = sanitize_for_artifact(original_line, max_chars=1600)
        if detect_pii_tags(sanitized_line, ignore_mask_tokens=True):
            sanitized_line = "[REDACTED_ORIGINAL_LINE]"
        payload["original_line"] = sanitized_line
    if complaint_id:
        payload["complaint_id"] = complaint_id
    if line_number is not None:
        payload["line_number"] = line_number

    with quarantine_path.open("a", encoding="utf-8") as outfile:
        outfile.write(json.dumps(payload, ensure_ascii=False))
        outfile.write("\n")


def _load_valid_raw_records(
    dataset_path: Path,
    quarantine_path: Path,
    sample_size: int | None,
) -> tuple[list[RawComplaint], LoaderStats]:
    valid_records: list[RawComplaint] = []
    counters = Counter()
    quarantined_by_reason: Counter[str] = Counter()

    quarantine_path.parent.mkdir(parents=True, exist_ok=True)
    quarantine_path.write_text("", encoding="utf-8")

    with dataset_path.open("r", encoding="utf-8") as infile:
        for line_number, raw_line in enumerate(infile, start=1):
            if sample_size is not None and counters["total"] >= sample_size:
                break

            counters["total"] += 1
            line = raw_line.strip()
            if not line:
                reason = REASON_JSON_PARSE_ERROR
                quarantined_by_reason[reason] += 1
                _append_quarantine_record(
                    quarantine_path=quarantine_path,
                    reason_code=reason,
                    error_message="Line is empty after trimming.",
                    schema_version=CONTRACT_VERSION,
                    original_line=raw_line,
                    line_number=line_number,
                )
                continue

            try:
                loaded = json.loads(line)
            except json.JSONDecodeError as error:
                reason = REASON_JSON_PARSE_ERROR
                quarantined_by_reason[reason] += 1
                _append_quarantine_record(
                    quarantine_path=quarantine_path,
                    reason_code=reason,
                    error_message=str(error),
                    schema_version=CONTRACT_VERSION,
                    original_line=line,
                    line_number=line_number,
                )
                continue

            if not isinstance(loaded, dict):
                reason = REASON_NON_OBJECT
                quarantined_by_reason[reason] += 1
                _append_quarantine_record(
                    quarantine_path=quarantine_path,
                    reason_code=reason,
                    error_message="JSON record is not an object.",
                    schema_version=CONTRACT_VERSION,
                    original_line=loaded if isinstance(loaded, str) else json.dumps(loaded, ensure_ascii=False),
                    line_number=line_number,
                )
                continue

            try:
                complaint = RawComplaint.model_validate(loaded)
            except ValidationError as error:
                reason = REASON_RAW_SCHEMA_INVALID
                quarantined_by_reason[reason] += 1
                _append_quarantine_record(
                    quarantine_path=quarantine_path,
                    reason_code=reason,
                    error_message=json.dumps(error.errors(include_input=False, include_url=False), ensure_ascii=False),
                    schema_version=CONTRACT_VERSION,
                    original_line=line,
                    line_number=line_number,
                )
                continue

            valid_records.append(complaint)
            counters["valid"] += 1

    return valid_records, LoaderStats(
        total=counters["total"],
        valid=counters["valid"],
        quarantined_by_reason=quarantined_by_reason,
    )


def _build_stable_complaint_id(record: RawComplaint, line_number_hint: int) -> str:
    if record.complaint_id:
        stripped = record.complaint_id.strip()
        if stripped:
            return stripped

    payload = "|".join(
        [
            record.brand_slug,
            record.created_at_iso or "",
            record.title.strip(),
            record.complaint_text.strip(),
            str(line_number_hint),
        ]
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return f"generated_{digest}"


def _truncate_at_word_boundary(text: str, max_chars: int) -> tuple[str, bool]:
    if len(text) <= max_chars:
        return text, False
    window = text[: max_chars + 1]
    split = window.rfind(" ")
    if split < int(max_chars * 0.7):
        split = max_chars
    truncated = text[:split].rstrip(" ,.;:-")
    return truncated, True


def _histogram(lengths: list[int]) -> dict[str, int]:
    buckets = {
        "0-79": 0,
        "80-199": 0,
        "200-499": 0,
        "500-999": 0,
        "1000-1999": 0,
        "2000-3999": 0,
        "4000-5999": 0,
        "6000+": 0,
    }
    for value in lengths:
        if value <= 79:
            buckets["0-79"] += 1
        elif value <= 199:
            buckets["80-199"] += 1
        elif value <= 499:
            buckets["200-499"] += 1
        elif value <= 999:
            buckets["500-999"] += 1
        elif value <= 1999:
            buckets["1000-1999"] += 1
        elif value <= 3999:
            buckets["2000-3999"] += 1
        elif value <= 5999:
            buckets["4000-5999"] += 1
        else:
            buckets["6000+"] += 1
    return buckets


def _compute_length_stats(lengths: list[int]) -> dict[str, Any]:
    if not lengths:
        return {
            "count": 0,
            "mean": 0.0,
            "median": 0.0,
            "p95": 0,
            "max": 0,
            "histogram_buckets": _histogram([]),
        }
    sorted_lengths = sorted(lengths)
    p95_index = max(0, math.ceil(0.95 * len(sorted_lengths)) - 1)
    return {
        "count": len(sorted_lengths),
        "mean": round(statistics.mean(sorted_lengths), 4),
        "median": float(statistics.median(sorted_lengths)),
        "p95": int(sorted_lengths[p95_index]),
        "max": int(sorted_lengths[-1]),
        "histogram_buckets": _histogram(sorted_lengths),
    }


def _resolve_multi_strategy(mode: str, configured: str) -> str:
    if configured != "auto":
        return configured
    return "truncate_primary" if mode == "SMOKE" else "split"


def _apply_duplicate_policy(
    records: list[CleanComplaint],
    mode: str,
    duplicate_cfg: dict[str, Any],
    seed: int,
) -> tuple[list[CleanComplaint], dict[str, Any], list[DuplicateCluster]]:
    if not duplicate_cfg.get("enabled", True) or not records:
        stats = {
            "enabled": False,
            "number_of_clusters": 0,
            "cluster_sizes": [],
            "duplicate_percentage": 0.0,
            "dropped_duplicates": 0,
        }
        return sorted(records, key=lambda item: item.complaint_id), stats, []

    complaint_ids = [record.complaint_id for record in records]
    texts = [record.complaint_text_clean for record in records]
    clusters = cluster_near_duplicates(
        complaint_ids=complaint_ids,
        texts=texts,
        shingle_size=int(duplicate_cfg["shingle_size"]),
        num_perm=int(duplicate_cfg["num_perm"]),
        bands=int(duplicate_cfg["bands"]),
        similarity_threshold=float(duplicate_cfg["similarity_threshold"]),
        random_seed=seed,
    )

    updated: dict[str, CleanComplaint] = {record.complaint_id: record for record in records}
    keep_ids = set(updated.keys())
    dropped_duplicates = 0

    for cluster in clusters:
        cluster_size = len(cluster.member_ids)
        for member_id in cluster.member_ids:
            record = updated[member_id]
            flags = sorted(set(record.quality_flags) | {"NEAR_DUPLICATE"})
            update_payload: dict[str, Any] = {
                "quality_flags": flags,
                "duplicate_cluster_id": cluster.cluster_id,
                "is_duplicate_of": None if member_id == cluster.canonical_id else cluster.canonical_id,
            }
            updated[member_id] = record.model_copy(update=update_payload)

        if mode == "FULL" and duplicate_cfg.get("full_mode_drop_duplicates", True):
            for member_id in cluster.member_ids:
                if member_id != cluster.canonical_id and member_id in keep_ids:
                    keep_ids.remove(member_id)
                    dropped_duplicates += 1
        elif mode == "SMOKE":
            keep_in_smoke = bool(duplicate_cfg.get("smoke_mode_keep_duplicates", True))
            drop_extreme = bool(duplicate_cfg.get("smoke_drop_extreme_clusters", False))
            extreme_threshold = int(duplicate_cfg.get("smoke_extreme_cluster_size", 25))
            if (not keep_in_smoke) or (drop_extreme and cluster_size >= extreme_threshold):
                for member_id in cluster.member_ids:
                    if member_id != cluster.canonical_id and member_id in keep_ids:
                        keep_ids.remove(member_id)
                        dropped_duplicates += 1

    final_records = [
        updated[record.complaint_id]
        for record in sorted(records, key=lambda item: item.complaint_id)
        if record.complaint_id in keep_ids
    ]
    duplicate_count = sum(max(0, len(cluster.member_ids) - 1) for cluster in clusters)
    duplicate_percentage = round((duplicate_count / len(records)) * 100.0, 4) if records else 0.0

    duplicate_stats = {
        "enabled": True,
        "number_of_clusters": len(clusters),
        "cluster_sizes": [len(cluster.member_ids) for cluster in clusters],
        "duplicate_percentage": duplicate_percentage,
        "dropped_duplicates": dropped_duplicates,
    }
    return final_records, duplicate_stats, clusters


def run_preprocess_stage(
    *,
    config: dict[str, Any],
    mode: str,
    logger: logging.Logger,
) -> dict[str, Any]:
    paths = config["paths"]
    preprocess_cfg = config["preprocess"]
    duplicate_cfg = preprocess_cfg["duplicates"]
    script_cfg = preprocess_cfg["script_noise"]
    min_chars = int(preprocess_cfg["min_chars"])
    max_chars = int(preprocess_cfg["max_chars"])
    sample_size = config.get("mode_runtime", {}).get("sample_size")
    run_timestamp = preprocess_cfg.get("output_timestamp_iso") or datetime.now(timezone.utc).isoformat()

    dataset_path = Path(paths["dataset"])
    clean_path = Path(paths["clean_complaints"])
    report_path = Path(paths["preprocess_report"])
    duplicates_path = Path(paths["duplicates_report"])
    quarantine_path = Path(paths["quarantine"])
    for output in [clean_path, report_path, duplicates_path, quarantine_path]:
        output.parent.mkdir(parents=True, exist_ok=True)

    records, loader_stats = _load_valid_raw_records(
        dataset_path=dataset_path,
        quarantine_path=quarantine_path,
        sample_size=sample_size,
    )
    log_event(
        logger,
        "INFO",
        "preprocess_loader_complete",
        {
            "total": loader_stats.total,
            "valid": loader_stats.valid,
            "quarantined_by_reason": dict(loader_stats.quarantined_by_reason),
        },
    )

    preprocess_reason_counts = Counter(loader_stats.quarantined_by_reason)
    cleaned_records: list[CleanComplaint] = []
    pii_detected_count = 0
    pii_masked_count = 0
    pii_leak_after_mask_count = 0
    script_noise_count = 0
    multi_complaint_suspected_count = 0
    too_short_count = 0
    too_long_truncated_count = 0

    multi_strategy = _resolve_multi_strategy(mode=mode, configured=preprocess_cfg["multi_complaint"]["strategy"])

    for index, raw_record in enumerate(records):
        complaint_id_base = _build_stable_complaint_id(raw_record, line_number_hint=index)
        try:
            title_clean, _ = clean_text_content(raw_record.title, indicators=DEFAULT_SCRIPT_INDICATORS)
            text_clean, _ = clean_text_content(raw_record.complaint_text, indicators=DEFAULT_SCRIPT_INDICATORS)

            script_assessment = assess_script_noise(
                raw_text=raw_record.complaint_text,
                cleaned_text=text_clean,
                indicators=script_cfg.get("indicators", DEFAULT_SCRIPT_INDICATORS),
                min_indicator_hits=int(script_cfg["min_indicator_hits"]),
                min_js_line_ratio=float(script_cfg["min_js_line_ratio"]),
                min_alpha_ratio=float(script_cfg["min_alpha_ratio"]),
                min_cleaned_ratio=float(script_cfg["min_cleaned_ratio"]),
            )
            if script_assessment.mostly_noise:
                preprocess_reason_counts[REASON_SCRIPT_NOISE] += 1
                script_noise_count += 1
                _append_quarantine_record(
                    quarantine_path=quarantine_path,
                    reason_code=REASON_SCRIPT_NOISE,
                    error_message=json.dumps(
                        {
                            "indicator_hits": script_assessment.indicator_hits,
                            "js_line_ratio": script_assessment.js_line_ratio,
                            "alpha_ratio": script_assessment.alpha_ratio,
                        }
                    ),
                    schema_version=CONTRACT_VERSION,
                    original_line=raw_record.complaint_text,
                    complaint_id=complaint_id_base,
                )
                continue

            segments = [text_clean]
            quality_flags = set(raw_record.quality_flags)
            multi_assessment = assess_multi_complaint(text=text_clean, brand_name=raw_record.brand_name)
            if multi_assessment.suspected:
                quality_flags.add("MULTI_COMPLAINT_SUSPECTED")
                multi_complaint_suspected_count += 1
                if multi_strategy == "quarantine":
                    preprocess_reason_counts[REASON_MULTI_COMPLAINT] += 1
                    _append_quarantine_record(
                        quarantine_path=quarantine_path,
                        reason_code=REASON_MULTI_COMPLAINT,
                        error_message="Multi-complaint contamination suspected and strategy is quarantine.",
                        schema_version=CONTRACT_VERSION,
                        original_line=text_clean,
                        complaint_id=complaint_id_base,
                    )
                    continue
                if multi_strategy == "split":
                    split_min_chars = int(preprocess_cfg["multi_complaint"]["split_min_chars"])
                    split_candidates = [item.strip() for item in multi_assessment.split_candidates if item.strip()]
                    segments = [item for item in split_candidates if len(item) >= split_min_chars]
                    if not segments:
                        segments = [
                            extract_primary_complaint(
                                text=text_clean,
                                brand_name=raw_record.brand_name,
                                min_chars=min_chars,
                            )
                        ]
                else:
                    segments = [
                        extract_primary_complaint(
                            text=text_clean,
                            brand_name=raw_record.brand_name,
                            min_chars=min_chars,
                        )
                    ]

            for segment_index, segment in enumerate(segments):
                segment_id = complaint_id_base if len(segments) == 1 else f"{complaint_id_base}-part{segment_index + 1}"
                segment_flags = set(quality_flags)

                masked_title = mask_pii_text(title_clean) if title_clean else mask_pii_text("")
                masked_text = mask_pii_text(segment)
                if masked_title.had_pii or masked_text.had_pii:
                    segment_flags.add("CONTAINED_PII_BEFORE_MASK")
                    pii_detected_count += 1
                    pii_masked_count += 1
                if masked_title.remaining_tags or masked_text.remaining_tags:
                    preprocess_reason_counts[REASON_PII_LEAK_AFTER_MASK] += 1
                    pii_leak_after_mask_count += 1
                    _append_quarantine_record(
                        quarantine_path=quarantine_path,
                        reason_code=REASON_PII_LEAK_AFTER_MASK,
                        error_message=json.dumps(
                            {
                                "remaining_title_tags": masked_title.remaining_tags,
                                "remaining_text_tags": masked_text.remaining_tags,
                            },
                            ensure_ascii=False,
                        ),
                        schema_version=CONTRACT_VERSION,
                        original_line=segment,
                        complaint_id=segment_id,
                    )
                    continue

                final_text, truncated = _truncate_at_word_boundary(masked_text.masked_text, max_chars=max_chars)
                if truncated:
                    segment_flags.add("TRUNCATED_LONG")
                    too_long_truncated_count += 1

                if len(final_text) < min_chars:
                    preprocess_reason_counts[REASON_TOO_SHORT] += 1
                    too_short_count += 1
                    _append_quarantine_record(
                        quarantine_path=quarantine_path,
                        reason_code=REASON_TOO_SHORT,
                        error_message=f"Text length {len(final_text)} is below minimum {min_chars}.",
                        schema_version=CONTRACT_VERSION,
                        original_line=final_text,
                        complaint_id=segment_id,
                    )
                    continue

                source_payload = "|".join([segment_id, raw_record.brand_slug, final_text, masked_title.masked_text])
                source_hash = hashlib.sha256(source_payload.encode("utf-8")).hexdigest()

                try:
                    clean_record = CleanComplaint(
                        complaint_id=segment_id,
                        brand_name=raw_record.brand_name,
                        brand_slug=raw_record.brand_slug,
                        created_at_iso=raw_record.created_at_iso,
                        normalized_category=raw_record.normalized_category,
                        original_category_label=raw_record.original_category_label,
                        title_clean=masked_title.masked_text if masked_title.masked_text else None,
                        complaint_text_clean=final_text,
                        tags=raw_record.tags,
                        support_count=raw_record.support_count,
                        quality_flags=sorted(segment_flags),
                        preprocess_version=preprocess_cfg["version"],
                        preprocess_timestamp_iso=run_timestamp,
                        source_hash_sha256=source_hash,
                    )
                except ValidationError as error:
                    preprocess_reason_counts[REASON_CLEAN_SCHEMA_INVALID] += 1
                    _append_quarantine_record(
                        quarantine_path=quarantine_path,
                        reason_code=REASON_CLEAN_SCHEMA_INVALID,
                        error_message=json.dumps(error.errors(include_input=False, include_url=False), ensure_ascii=False),
                        schema_version=CONTRACT_VERSION,
                        original_line=final_text,
                        complaint_id=segment_id,
                    )
                    continue

                cleaned_records.append(clean_record)

        except Exception as error:  # pragma: no cover - defensive isolation path
            preprocess_reason_counts[REASON_INTERNAL_ERROR] += 1
            _append_quarantine_record(
                quarantine_path=quarantine_path,
                reason_code=REASON_INTERNAL_ERROR,
                error_message=str(error),
                schema_version=CONTRACT_VERSION,
                original_line=f"record_index={index}",
                complaint_id=complaint_id_base,
            )
            if not config["pipeline"]["continue_on_record_error"]:
                raise

    deduped_records, duplicate_stats, clusters = _apply_duplicate_policy(
        records=cleaned_records,
        mode=mode,
        duplicate_cfg=duplicate_cfg,
        seed=int(config["reproducibility"]["seed"]),
    )

    with clean_path.open("w", encoding="utf-8") as outfile:
        for record in deduped_records:
            outfile.write(json.dumps(record.model_dump(mode="json"), ensure_ascii=False))
            outfile.write("\n")

    duplicates_report = {
        "mode": mode,
        "number_of_clusters": duplicate_stats["number_of_clusters"],
        "cluster_sizes": duplicate_stats["cluster_sizes"],
        "duplicate_percentage": duplicate_stats["duplicate_percentage"],
        "dropped_duplicates": duplicate_stats["dropped_duplicates"],
        "clusters": [
            {
                "cluster_id": cluster.cluster_id,
                "canonical_id": cluster.canonical_id,
                "member_ids": cluster.member_ids,
                "size": len(cluster.member_ids),
            }
            for cluster in clusters
        ],
    }
    duplicates_path.write_text(json.dumps(duplicates_report, ensure_ascii=False, indent=2), encoding="utf-8")

    lengths = [len(record.complaint_text_clean) for record in deduped_records]
    report = {
        "mode": mode,
        "preprocess_version": preprocess_cfg["version"],
        "total_records": loader_stats.total,
        "valid_records": len(deduped_records),
        "quarantined_records": int(sum(preprocess_reason_counts.values())),
        "quarantined_records_by_reason": dict(preprocess_reason_counts),
        "pii_detected_count": pii_detected_count,
        "pii_masked_count": pii_masked_count,
        "pii_leak_after_mask_count": pii_leak_after_mask_count,
        "script_noise_count": script_noise_count,
        "multi_complaint_suspected_count": multi_complaint_suspected_count,
        "too_short_count": too_short_count,
        "too_long_truncated_count": too_long_truncated_count,
        "duplicate_stats": duplicate_stats,
        "length_stats": _compute_length_stats(lengths),
        "loader_counters": {
            "total": loader_stats.total,
            "valid": loader_stats.valid,
            "quarantined_by_reason": dict(loader_stats.quarantined_by_reason),
        },
        "generated_at_utc": run_timestamp,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    log_event(
        logger,
        "INFO",
        "preprocess_stage_complete",
        {
            "valid_records": report["valid_records"],
            "quarantined_records": report["quarantined_records"],
            "duplicate_clusters": duplicate_stats["number_of_clusters"],
            "outputs": {
                "clean": str(clean_path),
                "report": str(report_path),
                "duplicates": str(duplicates_path),
                "quarantine": str(quarantine_path),
            },
        },
    )
    return report
