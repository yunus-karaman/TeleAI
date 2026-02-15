from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from data.schemas import CONTRACT_VERSION, NormalizedComplaint
from preprocess.pii import sanitize_for_artifact
from scripts.logging_utils import log_event
from solution_steps.generator import StepKBLink, generate_kb_and_links_for_steps, generate_solution_steps_for_category
from solution_steps.linting import lint_kb_paragraphs, lint_solution_steps
from solution_steps.patterns import mine_category_patterns
from solution_steps.quality import validate_solution_quality
from taxonomy.schema import TaxonomyCategory, load_taxonomy_file


def _load_jsonl_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Required input file not found: {path}")
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as infile:
        for line in infile:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _load_normalized_records(path: Path) -> list[NormalizedComplaint]:
    loaded = _load_jsonl_records(path)
    return [NormalizedComplaint.model_validate(item) for item in loaded]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, records: list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as outfile:
        for record in records:
            if hasattr(record, "model_dump"):
                payload = record.model_dump(mode="json")
            else:
                payload = record
            outfile.write(json.dumps(payload, ensure_ascii=False))
            outfile.write("\n")


def _append_quarantine(path: Path, reason_code: str, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "solution_steps",
        "reason_code": reason_code,
        "schema_version": CONTRACT_VERSION,
        "payload": sanitize_for_artifact(payload, max_chars=1800),
    }
    with path.open("a", encoding="utf-8") as outfile:
        outfile.write(json.dumps(entry, ensure_ascii=False))
        outfile.write("\n")


def _select_target_categories(
    taxonomy_categories: list[TaxonomyCategory],
    config: dict[str, Any],
    mode: str,
) -> list[str]:
    ordered_ids = [category.category_id for category in taxonomy_categories]
    if mode == "SMOKE":
        limit = int(config["smoke_category_limit"])
        return ordered_ids[: max(1, min(limit, len(ordered_ids)))]
    return ordered_ids


def _filter_smoke_violations(
    *,
    steps: list[Any],
    kb_items: list[Any],
    links: list[StepKBLink],
    step_lint: dict[str, Any],
    kb_lint: dict[str, Any],
    quarantine_path: Path,
) -> tuple[list[Any], list[Any], list[StepKBLink], dict[str, Any], dict[str, Any]]:
    bad_step_ids = {item["item_id"] for item in step_lint["violations"]}
    bad_paragraph_ids = {item["item_id"] for item in kb_lint["violations"]}
    if not bad_step_ids and not bad_paragraph_ids:
        return steps, kb_items, links, step_lint, kb_lint

    for violation in step_lint["violations"]:
        _append_quarantine(quarantine_path, reason_code="STEP_LINT_VIOLATION", payload=violation)
    for violation in kb_lint["violations"]:
        _append_quarantine(quarantine_path, reason_code="KB_LINT_VIOLATION", payload=violation)

    filtered_steps = [step for step in steps if step.step_id not in bad_step_ids]
    filtered_kb = [
        item
        for item in kb_items
        if item.paragraph_id not in bad_paragraph_ids and not any(step_id in bad_step_ids for step_id in item.applies_to_step_ids)
    ]
    valid_paragraph_ids = {item.paragraph_id for item in filtered_kb}
    filtered_links = []
    for link in links:
        if link.step_id in bad_step_ids:
            continue
        evidence_ids = [evidence_id for evidence_id in link.evidence_ids if evidence_id in valid_paragraph_ids]
        if not evidence_ids:
            _append_quarantine(
                quarantine_path,
                reason_code="LINK_REMOVED_AFTER_LINT",
                payload={"step_id": link.step_id, "original_evidence_ids": link.evidence_ids},
            )
            continue
        filtered_links.append(
            StepKBLink(
                step_id=link.step_id,
                evidence_ids=evidence_ids,
                rationale=link.rationale,
                version=link.version,
            )
        )

    return filtered_steps, filtered_kb, filtered_links, lint_solution_steps(filtered_steps), lint_kb_paragraphs(filtered_kb)


def run_solution_steps_stage(*, config: dict[str, Any], mode: str, logger: logging.Logger) -> dict[str, Any]:
    paths = config["paths"]
    stage_cfg = config["solution_steps"]
    taxonomy = load_taxonomy_file(stage_cfg["taxonomy_file"])
    taxonomy_map = {category.category_id: category for category in taxonomy.categories}

    labeled_path = Path(paths["labeled_complaints"])
    train_path = Path(paths["train_split"])
    val_path = Path(paths["val_split"])
    test_path = Path(paths["test_split"])
    hard_test_path = Path(paths["hard_test_split"])
    quarantine_path = Path(paths["quarantine"])

    labeled_records = _load_normalized_records(labeled_path)
    train_records = _load_normalized_records(train_path)
    val_records = _load_normalized_records(val_path)
    test_records = _load_normalized_records(test_path)
    hard_test_records = _load_normalized_records(hard_test_path)

    log_event(
        logger,
        "INFO",
        "solution_inputs_loaded",
        {
            "labeled_count": len(labeled_records),
            "train_count": len(train_records),
            "val_count": len(val_records),
            "test_count": len(test_records),
            "hard_test_count": len(hard_test_records),
        },
    )

    patterns = mine_category_patterns(
        train_records=train_records,
        taxonomy=taxonomy,
        top_k=int(stage_cfg["pattern_top_k"]),
    )
    _write_json(Path(paths["category_patterns"]), patterns)
    patterns_by_category = {item["category_id"]: item for item in patterns}

    target_categories = _select_target_categories(
        taxonomy_categories=taxonomy.categories,
        config=stage_cfg,
        mode=mode,
    )

    steps = []
    for category_id in target_categories:
        category_pattern = patterns_by_category.get(category_id, {"top_symptoms": [], "top_context_terms": [], "top_trigger_terms": []})
        steps.extend(
            generate_solution_steps_for_category(
                category_id=category_id,
                category_pattern=category_pattern,
                taxonomy_map=taxonomy_map,
                version=stage_cfg["version"],
            )
        )
    steps = sorted(steps, key=lambda item: item.step_id)

    kb_items, links = generate_kb_and_links_for_steps(
        steps=steps,
        version=stage_cfg["version"],
    )
    kb_items = sorted(kb_items, key=lambda item: item.paragraph_id)
    links = sorted(links, key=lambda item: item.step_id)

    step_lint = lint_solution_steps(steps)
    kb_lint = lint_kb_paragraphs(kb_items)

    if mode == "SMOKE" and (step_lint["violations_count"] > 0 or kb_lint["violations_count"] > 0):
        steps, kb_items, links, step_lint, kb_lint = _filter_smoke_violations(
            steps=steps,
            kb_items=kb_items,
            links=links,
            step_lint=step_lint,
            kb_lint=kb_lint,
            quarantine_path=quarantine_path,
        )

    _write_json(Path(paths["solution_step_lint_report"]), step_lint)
    _write_json(Path(paths["kb_lint_report"]), kb_lint)

    if mode == "FULL" and (step_lint["violations_count"] > 0 or kb_lint["violations_count"] > 0):
        raise RuntimeError("Lint violations detected in FULL mode. See lint reports.")

    quality = validate_solution_quality(
        steps=steps,
        kb_items=kb_items,
        links=links,
        target_categories=target_categories,
        config=stage_cfg["quality"],
    )
    if quality["errors"]:
        raise RuntimeError("Solution step quality checks failed: " + " | ".join(quality["errors"][:5]))

    _write_jsonl(Path(paths["solution_steps_jsonl"]), steps)
    _write_jsonl(Path(paths["kb_jsonl"]), kb_items)
    _write_jsonl(Path(paths["step_kb_links_jsonl"]), links)

    summary = {
        "mode": mode,
        "version": stage_cfg["version"],
        "target_categories": target_categories,
        "count_per_category": quality["count_per_category"],
        "count_per_level": quality["count_per_level"],
        "missing_evidence_count": quality["missing_evidence_count"],
        "safety_lint_violations_count": step_lint["violations_count"] + kb_lint["violations_count"],
        "determinism_hashes": quality["hashes"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(Path(paths["solution_steps_summary"]), summary)

    log_event(
        logger,
        "INFO",
        "solution_steps_stage_complete",
        {
            "steps": len(steps),
            "kb_paragraphs": len(kb_items),
            "links": len(links),
            "target_categories": len(target_categories),
            "outputs": {
                "patterns": paths["category_patterns"],
                "steps": paths["solution_steps_jsonl"],
                "kb": paths["kb_jsonl"],
                "links": paths["step_kb_links_jsonl"],
                "summary": paths["solution_steps_summary"],
            },
        },
    )
    return summary

