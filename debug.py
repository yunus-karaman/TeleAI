from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from data.schemas import CleanComplaint, GraphEdge, GraphNode, NormalizedComplaint
from evaluation.hallucination import evaluate_hallucination
from graph.retrieval import RetrievalResources, retrieve_evidence_pack
from models.template_renderer import render_deterministic_response
from preprocess.pii import detect_pii_tags, sanitize_for_artifact
from scripts.config_loader import load_config
from scripts.logging_utils import configure_json_logging, log_event
from scripts.reproducibility import set_global_determinism
from scripts.runtime_gates import handle_gate_violation
from scripts.solution_dataset_integrity import validate_solution_dataset
from training.data_builder import build_retrieval_resources_from_artifacts


@dataclass
class CheckResult:
    name: str
    status: str
    summary: str
    metrics: dict[str, Any]
    details: dict[str, Any]


def _status(value: bool) -> str:
    return "PASS" if value else "FAIL"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as infile:
        for line in infile:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _validate_jsonl_models(path: Path, model_type: Any, limit: int | None = None) -> tuple[int, int, list[str]]:
    rows = _read_jsonl(path)
    if limit is not None:
        rows = rows[: max(1, min(limit, len(rows)))]
    errors: list[str] = []
    for index, row in enumerate(rows, start=1):
        try:
            model_type.model_validate(row)
        except ValidationError as error:
            compact = json.dumps(error.errors(include_input=False, include_url=False), ensure_ascii=False)
            errors.append(f"{path}:{index}:{compact}")
    return len(rows), len(errors), errors[:8]


def _check_config(config: dict[str, Any], mode: str) -> CheckResult:
    required_fail_fast_keys = [
        "schema_violation",
        "pii_leak",
        "hallucination_violation",
        "missing_evidence",
        "graph_inconsistency",
    ]
    missing_or_false = [key for key in required_fail_fast_keys if not bool(config.get("fail_fast", {}).get(key, False))]
    trainer_cfg = config.get("training_llm", {}).get("trainer", {})
    invariant_ok = not bool(trainer_cfg.get("fallback_to_mock_on_failure", False))
    deterministic_ok = bool(config.get("reproducibility", {}).get("deterministic", False))
    seed = int(config.get("reproducibility", {}).get("seed", 0))
    ok = (len(missing_or_false) == 0) and invariant_ok and deterministic_ok and seed > 0
    return CheckResult(
        name="config_health",
        status=_status(ok),
        summary="Config, fail-fast and determinism settings checked.",
        metrics={
            "required_fail_fast_keys": len(required_fail_fast_keys),
            "missing_or_false_fail_fast_keys": len(missing_or_false),
            "deterministic": deterministic_ok,
            "seed": seed,
            "full_mode_mock_fallback_disabled": invariant_ok if mode == "FULL" else True,
        },
        details={
            "missing_or_false_fail_fast_keys": missing_or_false,
            "mode": mode,
        },
    )


def _check_dataset_schema(config: dict[str, Any], mode: str) -> CheckResult:
    paths = config["paths"]
    clean_path = Path(paths["clean_complaints"])
    labeled_path = Path(paths["labeled_complaints"])
    if not clean_path.exists() or not labeled_path.exists():
        missing = [str(path) for path in [clean_path, labeled_path] if not path.exists()]
        return CheckResult(
            name="data_health",
            status="FAIL",
            summary="Required dataset files missing.",
            metrics={"missing_files": len(missing)},
            details={"missing_files": missing},
        )

    sample_limit = None if mode == "FULL" else 4000
    clean_rows, clean_errors, clean_samples = _validate_jsonl_models(clean_path, CleanComplaint, sample_limit)
    labeled_rows, labeled_errors, labeled_samples = _validate_jsonl_models(labeled_path, NormalizedComplaint, sample_limit)
    ok = (clean_errors + labeled_errors) == 0
    return CheckResult(
        name="data_health",
        status=_status(ok),
        summary="Complaint artifact schema validation completed.",
        metrics={
            "clean_rows_checked": clean_rows,
            "labeled_rows_checked": labeled_rows,
            "clean_schema_errors": clean_errors,
            "labeled_schema_errors": labeled_errors,
        },
        details={"clean_schema_error_samples": clean_samples, "labeled_schema_error_samples": labeled_samples},
    )


def _check_pii(config: dict[str, Any], mode: str) -> CheckResult:
    paths = config["paths"]
    clean_rows = _read_jsonl(Path(paths["clean_complaints"]))
    labeled_rows = _read_jsonl(Path(paths["labeled_complaints"]))
    labeled_sample = labeled_rows[: min(600, len(labeled_rows))]

    blocking_tags = {"PHONE", "EMAIL", "IBAN", "TCKN", "DEVICE_ID", "ACCOUNT_ID"}
    clean_leaks: list[dict[str, Any]] = []
    clean_address_only = 0
    for row in clean_rows:
        text = f"{row.get('title_clean') or ''}\n{row.get('complaint_text_clean') or ''}"
        tags = detect_pii_tags(text, ignore_mask_tokens=True)
        blocking = sorted(set(tags) & blocking_tags)
        if blocking:
            clean_leaks.append(
                {
                    "complaint_id": row.get("complaint_id"),
                    "tags": blocking,
                    "sample": sanitize_for_artifact(text, max_chars=220),
                }
            )
        elif tags:
            clean_address_only += 1

    labeled_leaks: list[dict[str, Any]] = []
    labeled_address_only = 0
    for row in labeled_sample:
        text = f"{row.get('title_clean') or ''}\n{row.get('complaint_text_clean') or ''}"
        tags = detect_pii_tags(text, ignore_mask_tokens=True)
        blocking = sorted(set(tags) & blocking_tags)
        if blocking:
            labeled_leaks.append(
                {
                    "complaint_id": row.get("complaint_id"),
                    "tags": blocking,
                    "sample": sanitize_for_artifact(text, max_chars=220),
                }
            )
        elif tags:
            labeled_address_only += 1

    total_leaks = len(clean_leaks) + len(labeled_leaks)
    ok = total_leaks == 0
    return CheckResult(
        name="pii_scan",
        status=_status(ok),
        summary="PII leakage scan completed for clean artifacts and labeled sample.",
        metrics={
            "clean_rows_scanned": len(clean_rows),
            "labeled_rows_scanned": len(labeled_sample),
            "clean_blocking_pii_leak_count": len(clean_leaks),
            "labeled_blocking_pii_leak_count": len(labeled_leaks),
            "clean_address_only_flags": clean_address_only,
            "labeled_address_only_flags": labeled_address_only,
        },
        details={
            "clean_pii_samples": clean_leaks[:5],
            "labeled_pii_samples": labeled_leaks[:5],
            "mode": mode,
        },
    )


def _check_solution_dataset_integrity(config: dict[str, Any]) -> CheckResult:
    report = validate_solution_dataset(
        taxonomy_path=Path(config["taxonomy"]["taxonomy_file"]),
        solution_steps_path=Path(config["paths"]["solution_steps_jsonl"]),
        kb_path=Path(config["paths"]["kb_jsonl"]),
        step_kb_links_path=Path(config["paths"]["step_kb_links_jsonl"]),
        stage="debug_solution_dataset_integrity",
    )
    return CheckResult(
        name="solution_dataset_integrity",
        status=_status(bool(report["overall_pass"])),
        summary="Solution dataset integrity checks executed.",
        metrics=report["counts"],
        details={"violations": report["violations"][:15]},
    )


def _check_graph_integrity(config: dict[str, Any]) -> CheckResult:
    paths = config["paths"]
    nodes_path = Path(paths["graph_nodes"])
    edges_path = Path(paths["graph_edges"])

    if not nodes_path.exists() or not edges_path.exists():
        prerequisites = [
            Path(config["taxonomy"]["taxonomy_file"]),
            Path(paths["labeled_complaints"]),
            Path(paths["solution_steps_jsonl"]),
            Path(paths["kb_jsonl"]),
            Path(paths["step_kb_links_jsonl"]),
        ]
        missing = [str(path) for path in prerequisites if not path.exists()]
        ok = len(missing) == 0
        return CheckResult(
            name="graph_integrity",
            status=_status(ok),
            summary="Graph artifacts absent; build prerequisites checked.",
            metrics={"graph_artifacts_present": 0, "missing_prerequisites": len(missing)},
            details={"missing_prerequisites": missing},
        )

    node_rows = _read_jsonl(nodes_path)
    edge_rows = _read_jsonl(edges_path)
    node_errors: list[str] = []
    edge_errors: list[str] = []
    nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []

    for index, row in enumerate(node_rows, start=1):
        try:
            nodes.append(GraphNode.model_validate(row))
        except ValidationError as error:
            node_errors.append(f"nodes:{index}:{error}")
    for index, row in enumerate(edge_rows, start=1):
        try:
            edges.append(GraphEdge.model_validate(row))
        except ValidationError as error:
            edge_errors.append(f"edges:{index}:{error}")

    node_ids = {item.node_id for item in nodes}
    dangling_edges = [
        edge.edge_id
        for edge in edges
        if edge.source_node_id not in node_ids or edge.target_node_id not in node_ids
    ]
    degree: dict[str, int] = {node.node_id: 0 for node in nodes}
    for edge in edges:
        degree[edge.source_node_id] = degree.get(edge.source_node_id, 0) + 1
        degree[edge.target_node_id] = degree.get(edge.target_node_id, 0) + 1
    isolated_steps = [node.node_id for node in nodes if node.node_type == "solution_step" and degree.get(node.node_id, 0) == 0]
    missing_step_features = [
        node.node_id
        for node in nodes
        if node.node_type == "solution_step" and ("category_id" not in node.attributes or "level" not in node.attributes)
    ]

    ok = not node_errors and not edge_errors and not dangling_edges
    return CheckResult(
        name="graph_integrity",
        status=_status(ok),
        summary="Graph artifact schema and reference integrity validated.",
        metrics={
            "nodes": len(nodes),
            "edges": len(edges),
            "node_schema_errors": len(node_errors),
            "edge_schema_errors": len(edge_errors),
            "dangling_edges": len(dangling_edges),
            "isolated_solution_steps": len(isolated_steps),
            "missing_solution_step_features": len(missing_step_features),
        },
        details={
            "node_schema_error_samples": node_errors[:5],
            "edge_schema_error_samples": edge_errors[:5],
            "dangling_edge_samples": dangling_edges[:10],
            "isolated_solution_step_samples": isolated_steps[:10],
            "missing_solution_step_feature_samples": missing_step_features[:10],
        },
    )


def _check_retrieval(config: dict[str, Any], mode: str, logger: logging.Logger) -> tuple[CheckResult, RetrievalResources | None, list[NormalizedComplaint]]:
    try:
        resources, _, _, test_records, _ = build_retrieval_resources_from_artifacts(config=config, mode=mode, logger=logger)
    except Exception as error:  # noqa: BLE001
        return (
            CheckResult(
                name="retrieval_sanity",
                status="FAIL",
                summary="Retrieval resources could not be built.",
                metrics={"errors": 1},
                details={"error": str(error)},
            ),
            None,
            [],
        )

    sample = sorted(test_records, key=lambda item: item.complaint_id)[: min(5, len(test_records))]
    step_library_ids = {step.step_id for step in resources.steps}
    kb_library_ids = {kb.paragraph_id for kb in resources.kb_items}
    failures: list[str] = []

    for record in sample:
        pack, _ = retrieve_evidence_pack(
            complaint_text=record.complaint_text_clean,
            resources=resources,
            request_id=f"DEBUG:{record.complaint_id}",
            brand=record.brand_slug,
            include_debug=True,
        )
        evidence_ids = {item.paragraph_id for item in pack.evidence}
        for step in pack.top_steps:
            if not step.evidence_ids:
                failures.append(f"{record.complaint_id}:{step.step_id}:missing_evidence")
            if step.step_id not in step_library_ids:
                failures.append(f"{record.complaint_id}:{step.step_id}:step_not_in_library")
            for evidence_id in step.evidence_ids:
                if evidence_id not in evidence_ids:
                    failures.append(f"{record.complaint_id}:{step.step_id}:{evidence_id}:not_in_pack")
                if evidence_id not in kb_library_ids:
                    failures.append(f"{record.complaint_id}:{step.step_id}:{evidence_id}:not_in_library")

    deterministic_ok = True
    if sample:
        first = sample[0]
        pack_a, _ = retrieve_evidence_pack(
            complaint_text=first.complaint_text_clean,
            resources=resources,
            request_id=f"DEBUG:DET:{first.complaint_id}:A",
            brand=first.brand_slug,
            include_debug=False,
        )
        pack_b, _ = retrieve_evidence_pack(
            complaint_text=first.complaint_text_clean,
            resources=resources,
            request_id=f"DEBUG:DET:{first.complaint_id}:B",
            brand=first.brand_slug,
            include_debug=False,
        )
        deterministic_ok = [item.step_id for item in pack_a.top_steps] == [item.step_id for item in pack_b.top_steps]
        deterministic_ok = deterministic_ok and [item.paragraph_id for item in pack_a.evidence] == [
            item.paragraph_id for item in pack_b.evidence
        ]
        if not deterministic_ok:
            failures.append("determinism_failed_for_first_sample")

    ok = (len(failures) == 0) and deterministic_ok
    return (
        CheckResult(
            name="retrieval_sanity",
            status=_status(ok),
            summary="EvidencePack retrieval consistency and determinism validated.",
            metrics={
                "sample_size": len(sample),
                "library_steps": len(step_library_ids),
                "library_evidence": len(kb_library_ids),
                "failure_count": len(failures),
                "deterministic": deterministic_ok,
            },
            details={"failure_samples": failures[:15]},
        ),
        resources,
        sample,
    )


def _check_hallucination_sanity(
    config: dict[str, Any],
    resources: RetrievalResources | None,
    sample_records: list[NormalizedComplaint],
) -> CheckResult:
    if resources is None or not sample_records:
        return CheckResult(
            name="hallucination_compliance",
            status="FAIL",
            summary="No retrieval sample available for hallucination sanity check.",
            metrics={"cases": 0},
            details={},
        )

    record = sample_records[0]
    pack, _ = retrieve_evidence_pack(
        complaint_text=record.complaint_text_clean,
        resources=resources,
        request_id=f"DEBUG:HALL:{record.complaint_id}",
        brand=record.brand_slug,
        include_debug=False,
    )
    response_text = render_deterministic_response(
        pack,
        min_steps=int(config["training_llm"]["dataset"]["task2_min_steps"]),
        max_steps=int(config["training_llm"]["dataset"]["task2_top_steps"]),
    )
    case = {
        "split": "debug",
        "complaint_id": record.complaint_id,
        "complaint_text_clean": record.complaint_text_clean,
        "true_category": record.normalized_category,
        "inference": {
            "response_text": response_text,
            "validation": {"template_compliant": True},
            "evidence_pack": pack.model_dump(mode="json"),
        },
    }

    debug_dir = Path(config["paths"].get("debug_dir", "artifacts/debug"))
    hall_json = debug_dir / "hallucination_sanity_report.json"
    hall_md = debug_dir / "hallucination_sanity_report.md"
    report = evaluate_hallucination(
        inference_cases=[case],
        report_json_path=hall_json,
        report_md_path=hall_md,
    )
    ok = (
        report["metrics"]["step_hallucination_rate"] == 0.0
        and report["metrics"]["citation_hallucination_rate"] == 0.0
        and report["metrics"]["hallucination_rate_actionable"] == 0.0
    )
    return CheckResult(
        name="hallucination_compliance",
        status=_status(ok),
        summary="Hallucination scorer sanity check on deterministic template output.",
        metrics={
            "cases": report["counts"]["cases"],
            "hallucination_rate_actionable": report["metrics"]["hallucination_rate_actionable"],
            "step_hallucination_rate": report["metrics"]["step_hallucination_rate"],
            "citation_hallucination_rate": report["metrics"]["citation_hallucination_rate"],
            "evidence_mismatch_rate": report["metrics"]["evidence_mismatch_rate"],
        },
        details={"report_json": str(hall_json), "report_md": str(hall_md)},
    )


def _write_debug_reports(config: dict[str, Any], payload: dict[str, Any]) -> tuple[Path, Path]:
    debug_dir = Path(config["paths"].get("debug_dir", "artifacts/debug"))
    debug_dir.mkdir(parents=True, exist_ok=True)
    json_path = debug_dir / "debug_report.json"
    md_path = debug_dir / "debug_report.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Debug Report",
        "",
        f"- mode: `{payload['mode']}`",
        f"- check: `{payload['check']}`",
        f"- overall_status: `{payload['overall_status']}`",
        "",
        "## Module Status",
    ]
    for row in payload["modules"]:
        lines.append(f"- {row['name']}: `{row['status']}`")
    lines.append("")
    lines.append("## Failed Modules")
    if not payload["failed_modules"]:
        lines.append("- none")
    else:
        for name in payload["failed_modules"]:
            lines.append(f"- {name}")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path


def run_debug(*, config_path: str, mode: str, check: str) -> int:
    if check != "all":
        raise ValueError("Only '--check all' is supported.")

    config = load_config(config_path=config_path, mode=mode)
    logger = configure_json_logging(level=config["logging"]["level"], log_file=config["logging"]["file"])
    set_global_determinism(
        seed=int(config["reproducibility"]["seed"]),
        deterministic=bool(config["reproducibility"]["deterministic"]),
    )

    log_event(logger, "INFO", "debug_start", {"mode": mode, "check": check})

    results: list[CheckResult] = []
    results.append(_check_config(config, mode))
    results.append(_check_dataset_schema(config, mode))
    results.append(_check_pii(config, mode))
    results.append(_check_solution_dataset_integrity(config))
    results.append(_check_graph_integrity(config))
    retrieval_result, resources, sample_records = _check_retrieval(config, mode, logger)
    results.append(retrieval_result)
    results.append(_check_hallucination_sanity(config, resources, sample_records))

    failed_modules = [item.name for item in results if item.status == "FAIL"]
    overall_status = "PASS" if not failed_modules else "FAIL"
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "check": check,
        "overall_status": overall_status,
        "failed_modules": failed_modules,
        "modules": [
            {
                "name": item.name,
                "status": item.status,
                "summary": item.summary,
                "metrics": item.metrics,
                "details": item.details,
            }
            for item in results
        ],
    }

    report_json, report_md = _write_debug_reports(config, payload)
    log_event(
        logger,
        "INFO",
        "debug_complete",
        {"overall_status": overall_status, "failed_modules": failed_modules, "report_json": str(report_json)},
    )

    if failed_modules:
        gate_key = "schema_violation"
        if "pii_scan" in failed_modules:
            gate_key = "pii_leak"
        elif "hallucination_compliance" in failed_modules:
            gate_key = "hallucination_violation"
        elif "graph_integrity" in failed_modules:
            gate_key = "graph_inconsistency"
        elif "retrieval_sanity" in failed_modules:
            gate_key = "missing_evidence"
        handle_gate_violation(
            config=config,
            mode=mode,
            stage="debug",
            gate_key=gate_key,
            reason_code="DEBUG_CHECK_FAILED",
            message=f"Debug harness failed modules: {', '.join(failed_modules)}",
            details={"failed_modules": failed_modules, "debug_report_json": str(report_json), "debug_report_md": str(report_md)},
            logger=logger,
        )
        if mode == "FULL":
            return 1

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Project debug harness")
    parser.add_argument("--check", default="all")
    parser.add_argument("--mode", choices=["SMOKE", "FULL"], default="FULL")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    return run_debug(config_path=args.config, mode=args.mode, check=args.check)


if __name__ == "__main__":
    raise SystemExit(main())
