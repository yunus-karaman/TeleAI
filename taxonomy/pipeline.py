from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from data.schemas import CONTRACT_VERSION, CleanComplaint, NormalizedComplaint
from preprocess.pii import sanitize_for_artifact
from scripts.logging_utils import log_event
from taxonomy.assignment import HybridTaxonomyAssigner
from taxonomy.baselines import BaselineEvaluation, run_baselines
from taxonomy.reporting import build_taxonomy_report, export_error_analysis, write_taxonomy_markdown
from taxonomy.schema import load_taxonomy_file
from taxonomy.splitting import SplitResult, create_splits


def _append_quarantine(
    quarantine_path: Path,
    reason_code: str,
    error_message: str,
    record_payload: Any,
    line_number: int | None = None,
    complaint_id: str | None = None,
) -> None:
    quarantine_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "taxonomy",
        "reason_code": reason_code,
        "schema_version": CONTRACT_VERSION,
        "error_message": sanitize_for_artifact(error_message, max_chars=500),
        "record": sanitize_for_artifact(record_payload, max_chars=1600),
    }
    if line_number is not None:
        payload["line_number"] = line_number
    if complaint_id is not None:
        payload["complaint_id"] = complaint_id
    with quarantine_path.open("a", encoding="utf-8") as out:
        out.write(json.dumps(payload, ensure_ascii=False))
        out.write("\n")


def _load_clean_records(
    clean_path: Path,
    quarantine_path: Path,
    sample_size: int | None,
) -> tuple[list[CleanComplaint], Counter[str]]:
    records: list[CleanComplaint] = []
    reason_counts: Counter[str] = Counter()

    with clean_path.open("r", encoding="utf-8") as infile:
        for line_number, raw_line in enumerate(infile, start=1):
            if sample_size is not None and len(records) >= sample_size:
                break
            line = raw_line.strip()
            if not line:
                reason_counts["EMPTY_LINE"] += 1
                _append_quarantine(
                    quarantine_path,
                    reason_code="EMPTY_LINE",
                    error_message="Empty line encountered in complaints_clean.jsonl",
                    record_payload=raw_line,
                    line_number=line_number,
                )
                continue
            try:
                loaded = json.loads(line)
            except json.JSONDecodeError as error:
                reason_counts["JSON_PARSE_ERROR"] += 1
                _append_quarantine(
                    quarantine_path,
                    reason_code="JSON_PARSE_ERROR",
                    error_message=str(error),
                    record_payload=line,
                    line_number=line_number,
                )
                continue

            try:
                record = CleanComplaint.model_validate(loaded)
            except ValidationError as error:
                reason_counts["CLEAN_SCHEMA_INVALID"] += 1
                _append_quarantine(
                    quarantine_path,
                    reason_code="CLEAN_SCHEMA_INVALID",
                    error_message=json.dumps(error.errors(include_url=False, include_input=False), ensure_ascii=False),
                    record_payload=loaded,
                    line_number=line_number,
                    complaint_id=loaded.get("complaint_id") if isinstance(loaded, dict) else None,
                )
                continue

            records.append(record)

    return records, reason_counts


def _write_jsonl(path: Path, records: list[NormalizedComplaint]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as out:
        for record in records:
            out.write(json.dumps(record.model_dump(mode="json"), ensure_ascii=False))
            out.write("\n")


def _split_to_paths(paths_cfg: dict[str, Any], split_result: SplitResult) -> None:
    _write_jsonl(Path(paths_cfg["train_split"]), split_result.train)
    _write_jsonl(Path(paths_cfg["val_split"]), split_result.val)
    _write_jsonl(Path(paths_cfg["test_split"]), split_result.test)
    _write_jsonl(Path(paths_cfg["hard_test_split"]), split_result.hard_test)


def _count_duplicate_leakage(split_result: SplitResult) -> int:
    split_by_cluster: dict[str, set[str]] = defaultdict(set)
    for split_name, records in {
        "train": split_result.train,
        "val": split_result.val,
        "test": split_result.test,
    }.items():
        for record in records:
            cluster = record.duplicate_cluster_id
            if cluster:
                split_by_cluster[cluster].add(split_name)
    return sum(1 for splits in split_by_cluster.values() if len(splits) > 1)


def _create_normalized_record(clean_record: CleanComplaint, assignment: Any, taxonomy_version: str) -> NormalizedComplaint:
    flags = set(clean_record.quality_flags)
    if assignment.needs_review:
        flags.add("NEEDS_REVIEW")
    if assignment.normalized_category == "OTHER" and clean_record.normalized_category != "OTHER":
        flags.add("TAXONOMY_OTHER_FALLBACK")

    return NormalizedComplaint(
        complaint_id=clean_record.complaint_id,
        brand_name=clean_record.brand_name,
        brand_slug=clean_record.brand_slug,
        created_at_iso=clean_record.created_at_iso,
        title_clean=clean_record.title_clean,
        complaint_text_clean=clean_record.complaint_text_clean,
        normalized_category=assignment.normalized_category,
        confidence_score=assignment.confidence_score,
        assignment_reason=assignment.assignment_reason,
        needs_review=assignment.needs_review,
        source_category=clean_record.normalized_category,
        quality_flags=sorted(flags),
        duplicate_cluster_id=clean_record.duplicate_cluster_id,
        is_duplicate_of=clean_record.is_duplicate_of,
        taxonomy_version=taxonomy_version,
        source_hash_sha256=clean_record.source_hash_sha256,
    )


def run_taxonomy_stage(*, config: dict[str, Any], mode: str, logger: logging.Logger) -> dict[str, Any]:
    paths = config["paths"]
    taxonomy_cfg = config["taxonomy"]
    taxonomy_file_path = taxonomy_cfg["taxonomy_file"]
    taxonomy = load_taxonomy_file(taxonomy_file_path)

    sample_size = config.get("mode_runtime", {}).get("sample_size")
    clean_records, quarantine_counts = _load_clean_records(
        clean_path=Path(paths["clean_complaints"]),
        quarantine_path=Path(paths["quarantine"]),
        sample_size=sample_size,
    )
    log_event(
        logger,
        "INFO",
        "taxonomy_clean_load_complete",
        {
            "loaded_records": len(clean_records),
            "quarantined_by_reason": dict(quarantine_counts),
        },
    )
    if not clean_records:
        raise RuntimeError("No valid clean records available for taxonomy stage.")

    assigner = HybridTaxonomyAssigner(
        taxonomy=taxonomy,
        config=taxonomy_cfg["assignment"],
        seed=int(config["reproducibility"]["seed"]),
    )
    assigner.fit([record.complaint_text_clean for record in clean_records])

    labeled_records: list[NormalizedComplaint] = []
    assignment_debug: list[dict[str, Any]] = []
    for clean_record in clean_records:
        assignment = assigner.assign(clean_record.complaint_text_clean)
        labeled = _create_normalized_record(
            clean_record=clean_record,
            assignment=assignment,
            taxonomy_version=taxonomy.taxonomy_version,
        )
        labeled_records.append(labeled)
        assignment_debug.append(
            {
                "complaint_id": labeled.complaint_id,
                "final_category": labeled.normalized_category,
                "confidence_score": labeled.confidence_score,
                "rule_top_category": assignment.rule_top_category,
                "embedding_top_category": assignment.embedding_top_category,
                "assignment_reason": labeled.assignment_reason,
                "text_snippet": labeled.complaint_text_clean[:280],
            }
        )

    labeled_records = sorted(labeled_records, key=lambda item: item.complaint_id)
    _write_jsonl(Path(paths["labeled_complaints"]), labeled_records)

    split_result = create_splits(records=labeled_records, split_config=taxonomy_cfg["splits"])
    _split_to_paths(paths_cfg=paths, split_result=split_result)
    duplicate_leak_count = _count_duplicate_leakage(split_result)

    baseline_results = run_baselines(
        train_records=split_result.train,
        test_records=split_result.test,
        hard_test_records=split_result.hard_test,
        config=taxonomy_cfg["baselines"],
        mode=mode,
        seed=int(config["reproducibility"]["seed"]),
    )

    baseline_primary: BaselineEvaluation | None = baseline_results.get("baseline_tfidf_linear")
    error_exports = export_error_analysis(
        output_dir=paths["taxonomy_error_analysis_dir"],
        test_records=split_result.test,
        baseline_result=baseline_primary,
        assignment_debug=assignment_debug,
        top_n=int(taxonomy_cfg["error_analysis"]["top_n_per_class"]),
    )

    split_counts = {
        "train": len(split_result.train),
        "val": len(split_result.val),
        "test": len(split_result.test),
        "hard_test": len(split_result.hard_test),
    }
    report = build_taxonomy_report(
        taxonomy=taxonomy,
        labeled_records=labeled_records,
        split_counts=split_counts,
        duplicate_cluster_cross_split_count=duplicate_leak_count,
        baseline_results=baseline_results,
        needs_review_sample_size=int(taxonomy_cfg["report"]["needs_review_sample_size"]),
        assignment_debug=assignment_debug,
        test_records=split_result.test,
        error_export_paths=error_exports,
    )
    report["quarantined_by_reason"] = dict(quarantine_counts)
    report["generated_at_utc"] = datetime.now(timezone.utc).isoformat()

    report_path = Path(paths["taxonomy_report_json"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_taxonomy_markdown(report, output_path=paths["taxonomy_report_md"])

    log_event(
        logger,
        "INFO",
        "taxonomy_stage_complete",
        {
            "labeled_records": len(labeled_records),
            "split_counts": split_counts,
            "duplicate_cluster_cross_split_count": duplicate_leak_count,
            "outputs": {
                "taxonomy_yaml": taxonomy_file_path,
                "labeled": paths["labeled_complaints"],
                "train": paths["train_split"],
                "val": paths["val_split"],
                "test": paths["test_split"],
                "hard_test": paths["hard_test_split"],
                "report_json": paths["taxonomy_report_json"],
                "report_md": paths["taxonomy_report_md"],
            },
        },
    )
    return report

