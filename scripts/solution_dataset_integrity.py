from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from data.schemas import KBParagraph, SolutionStep
from scripts.runtime_gates import handle_gate_violation
from solution_steps.generator import StepKBLink
from taxonomy.schema import TaxonomyFile, load_taxonomy_file


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as infile:
        for line in infile:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _load_jsonl_models(path: Path, model_type: Any) -> tuple[list[Any], list[str]]:
    rows: list[Any] = []
    errors: list[str] = []
    if not path.exists():
        errors.append(f"missing_file:{path}")
        return rows, errors

    try:
        records = _read_jsonl(path)
    except Exception as error:  # noqa: BLE001
        errors.append(f"read_error:{path}:{error}")
        return rows, errors

    for index, payload in enumerate(records, start=1):
        try:
            rows.append(model_type.model_validate(payload))
        except ValidationError as error:
            compact = json.dumps(error.errors(include_input=False, include_url=False), ensure_ascii=False)
            errors.append(f"schema_error:{path}:{index}:{compact}")
    return rows, errors


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_md(path: Path, report: dict[str, Any]) -> None:
    lines: list[str] = [
        "# Solution Dataset Integrity Report",
        "",
        f"- overall_pass: `{report['overall_pass']}`",
        f"- stage: `{report['stage']}`",
        "",
        "## Counts",
    ]
    for key, value in report["counts"].items():
        lines.append(f"- {key}: `{value}`")
    lines.append("")
    lines.append("## Violations")
    if not report["violations"]:
        lines.append("- none")
    else:
        for item in report["violations"]:
            lines.append(f"- [{item['severity']}] {item['code']}: {item['message']}")
            if item.get("samples"):
                for sample in item["samples"][:5]:
                    lines.append(f"  - sample: `{sample}`")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_solution_dataset(
    *,
    taxonomy_path: Path,
    solution_steps_path: Path,
    kb_path: Path,
    step_kb_links_path: Path,
    stage: str = "solution_dataset_integrity",
) -> dict[str, Any]:
    violations: list[dict[str, Any]] = []

    taxonomy: TaxonomyFile | None = None
    taxonomy_error: str | None = None
    try:
        taxonomy = load_taxonomy_file(taxonomy_path)
    except Exception as error:  # noqa: BLE001
        taxonomy_error = str(error)
        violations.append(
            {
                "severity": "P0",
                "code": "TAXONOMY_INVALID",
                "message": f"taxonomy could not be loaded: {error}",
                "samples": [str(taxonomy_path)],
            }
        )

    steps, step_schema_errors = _load_jsonl_models(solution_steps_path, SolutionStep)
    kb_rows, kb_schema_errors = _load_jsonl_models(kb_path, KBParagraph)
    links, link_schema_errors = _load_jsonl_models(step_kb_links_path, StepKBLink)

    if step_schema_errors:
        violations.append(
            {
                "severity": "P0",
                "code": "SOLUTION_STEP_SCHEMA_INVALID",
                "message": "solution step schema validation failed",
                "samples": step_schema_errors[:10],
            }
        )
    if kb_schema_errors:
        violations.append(
            {
                "severity": "P0",
                "code": "KB_SCHEMA_INVALID",
                "message": "kb paragraph schema validation failed",
                "samples": kb_schema_errors[:10],
            }
        )
    if link_schema_errors:
        violations.append(
            {
                "severity": "P0",
                "code": "LINK_SCHEMA_INVALID",
                "message": "step-kb link schema validation failed",
                "samples": link_schema_errors[:10],
            }
        )

    step_ids = [item.step_id for item in steps]
    kb_ids = [item.paragraph_id for item in kb_rows]
    link_step_ids = [item.step_id for item in links]

    duplicate_step_ids = sorted({item for item in step_ids if step_ids.count(item) > 1})
    duplicate_kb_ids = sorted({item for item in kb_ids if kb_ids.count(item) > 1})
    duplicate_link_step_ids = sorted({item for item in link_step_ids if link_step_ids.count(item) > 1})

    if duplicate_step_ids:
        violations.append(
            {
                "severity": "P0",
                "code": "DUPLICATE_STEP_ID",
                "message": "duplicate step_id values detected",
                "samples": duplicate_step_ids[:10],
            }
        )
    if duplicate_kb_ids:
        violations.append(
            {
                "severity": "P0",
                "code": "DUPLICATE_KB_PARAGRAPH_ID",
                "message": "duplicate paragraph_id values detected",
                "samples": duplicate_kb_ids[:10],
            }
        )
    if duplicate_link_step_ids:
        violations.append(
            {
                "severity": "P1",
                "code": "DUPLICATE_LINK_STEP_ID",
                "message": "multiple link rows for same step_id detected",
                "samples": duplicate_link_step_ids[:10],
            }
        )

    taxonomy_categories = {item.category_id for item in taxonomy.categories} if taxonomy is not None else set()
    unknown_step_categories = sorted({item.category_id for item in steps if item.category_id not in taxonomy_categories})
    if unknown_step_categories:
        violations.append(
            {
                "severity": "P0",
                "code": "STEP_CATEGORY_NOT_IN_TAXONOMY",
                "message": "solution steps contain categories not present in taxonomy",
                "samples": unknown_step_categories[:10],
            }
        )

    step_id_set = set(step_ids)
    kb_id_set = set(kb_ids)
    link_step_set = set(link_step_ids)

    dangling_step_refs: list[str] = []
    dangling_evidence_refs: list[str] = []
    empty_evidence_links: list[str] = []

    for link in links:
        if link.step_id not in step_id_set:
            dangling_step_refs.append(link.step_id)
        if not link.evidence_ids:
            empty_evidence_links.append(link.step_id)
        for evidence_id in link.evidence_ids:
            if evidence_id not in kb_id_set:
                dangling_evidence_refs.append(f"{link.step_id}->{evidence_id}")

    steps_without_evidence = sorted(step_id_set - link_step_set)

    if dangling_step_refs:
        violations.append(
            {
                "severity": "P0",
                "code": "DANGLING_STEP_LINK",
                "message": "link rows reference unknown step_id",
                "samples": sorted(set(dangling_step_refs))[:10],
            }
        )
    if dangling_evidence_refs:
        violations.append(
            {
                "severity": "P0",
                "code": "DANGLING_EVIDENCE_LINK",
                "message": "link rows reference unknown evidence_id",
                "samples": sorted(set(dangling_evidence_refs))[:10],
            }
        )
    if empty_evidence_links:
        violations.append(
            {
                "severity": "P0",
                "code": "EMPTY_EVIDENCE_LINK",
                "message": "some link rows have zero evidence_ids",
                "samples": sorted(set(empty_evidence_links))[:10],
            }
        )
    if steps_without_evidence:
        violations.append(
            {
                "severity": "P0",
                "code": "STEP_WITHOUT_EVIDENCE",
                "message": "some steps are missing evidence links",
                "samples": steps_without_evidence[:10],
            }
        )

    counts = {
        "taxonomy_categories": len(taxonomy_categories),
        "solution_steps": len(steps),
        "kb_paragraphs": len(kb_rows),
        "step_kb_links": len(links),
        "step_schema_errors": len(step_schema_errors),
        "kb_schema_errors": len(kb_schema_errors),
        "link_schema_errors": len(link_schema_errors),
        "steps_without_evidence": len(steps_without_evidence),
        "dangling_step_refs": len(dangling_step_refs),
        "dangling_evidence_refs": len(dangling_evidence_refs),
    }

    overall_pass = len(violations) == 0
    return {
        "stage": stage,
        "overall_pass": overall_pass,
        "taxonomy_error": taxonomy_error,
        "counts": counts,
        "violations": violations,
        "paths": {
            "taxonomy": str(taxonomy_path),
            "solution_steps": str(solution_steps_path),
            "kb": str(kb_path),
            "step_kb_links": str(step_kb_links_path),
        },
    }


def run_solution_dataset_integrity(
    *,
    config: dict[str, Any],
    mode: str,
    logger: Any | None = None,
    stage: str = "solution_dataset_integrity",
) -> dict[str, Any]:
    taxonomy_path = Path(config["taxonomy"]["taxonomy_file"])
    steps_path = Path(config["paths"]["solution_steps_jsonl"])
    kb_path = Path(config["paths"]["kb_jsonl"])
    links_path = Path(config["paths"]["step_kb_links_jsonl"])

    report = validate_solution_dataset(
        taxonomy_path=taxonomy_path,
        solution_steps_path=steps_path,
        kb_path=kb_path,
        step_kb_links_path=links_path,
        stage=stage,
    )

    integrity_dir = Path(config["paths"].get("integrity_dir", "artifacts/integrity"))
    report_json = integrity_dir / "solution_dataset_integrity_report.json"
    report_md = integrity_dir / "solution_dataset_integrity_report.md"
    _write_json(report_json, report)
    _write_md(report_md, report)

    if not report["overall_pass"]:
        reason_code = "SOLUTION_DATASET_INTEGRITY_FAILED"
        message = "Solution dataset integrity checks failed."
        has_missing_evidence = any(item["code"] == "STEP_WITHOUT_EVIDENCE" for item in report["violations"])
        gate_key = "missing_evidence" if has_missing_evidence else "schema_violation"
        handle_gate_violation(
            config=config,
            mode=mode,
            stage=stage,
            gate_key=gate_key,
            reason_code=reason_code,
            message=message,
            details={"report_json": str(report_json), "violation_count": len(report["violations"])},
            logger=logger,
        )

    return report
