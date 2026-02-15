from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from api.chat_service import ChatService
from evaluation.common import load_normalized_jsonl, run_inference_records, write_json
from evaluation.hallucination import evaluate_hallucination
from evaluation.pii_leakage import evaluate_pii_leakage
from evaluation.security_adversarial import evaluate_security_adversarial
from evaluation.task_metrics import evaluate_task_metrics
from models.infer import ConstrainedInferenceEngine
from scripts.logging_utils import log_event
from scripts.runtime_gates import handle_gate_violation


def _select_records(records: list[Any], limit: int | None) -> list[Any]:
    if limit is None:
        return records
    return records[: max(1, min(limit, len(records)))]


def _update_task_report_with_chat_metric(
    *,
    task_report: dict[str, Any],
    engine: ConstrainedInferenceEngine,
    records: list[Any],
    max_attempts: int,
) -> dict[str, Any]:
    service = ChatService(engine=engine, max_attempts=max_attempts)
    simulated = _select_records(records, min(40, len(records)))
    escalated = 0
    for idx, record in enumerate(simulated, start=1):
        start_payload = service.start_session(record.complaint_text_clean)
        session_id = start_payload["session_id"]
        service.continue_session(session_id, "Hayır")
        state = service.get_state(session_id)
        if state["stage"] == "AWAITING_CLARIFICATION":
            service.continue_session(session_id, f"Ek test bilgisi {idx}")
        final = service.continue_session(session_id, "Hayır")
        if final["status"] == "ESCALATED":
            escalated += 1

    rate = (escalated / float(len(simulated))) if simulated else 0.0
    task_report.setdefault("escalation", {})
    task_report["escalation"]["repeated_failure_escalation_rate"] = round(rate, 6)
    return task_report


def _write_task_markdown(path: str | Path, report: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Task Metrics Report",
        "",
        f"- intent_accuracy: `{report['intent_metrics']['accuracy']}`",
        f"- intent_macro_f1: `{report['intent_metrics']['macro_f1']}`",
        f"- step_validity_rate: `{report['step_quality']['step_validity_rate']}`",
        f"- step_count_correctness_rate: `{report['step_quality']['step_count_correctness_rate']}`",
        f"- proxy_relevance_mean: `{report['step_quality']['proxy_relevance_mean']}`",
        f"- high_risk_consistency: `{report['escalation']['high_risk_consistency']}`",
        f"- repeated_failure_escalation_rate: `{report['escalation']['repeated_failure_escalation_rate']}`",
        f"- latency_p50_ms: `{report['performance']['latency_p50_ms']}`",
        f"- latency_p95_ms: `{report['performance']['latency_p95_ms']}`",
        f"- throughput_rps: `{report['performance']['throughput_rps']}`",
    ]
    target.write_text("\n".join(lines), encoding="utf-8")


def run_eval_stage(*, config: dict[str, Any], mode: str, logger: logging.Logger) -> dict[str, Any]:
    eval_cfg = config["evaluation"]
    paths = config["paths"]
    mode_cfg = eval_cfg["mode"]["SMOKE" if mode == "SMOKE" else "FULL"]

    test_records = load_normalized_jsonl(paths["test_split"])
    hard_records = load_normalized_jsonl(paths["hard_test_split"])
    test_records = _select_records(test_records, mode_cfg.get("test_limit"))
    hard_records = _select_records(hard_records, mode_cfg.get("hard_test_limit"))

    engine = ConstrainedInferenceEngine(config=config, mode=mode, logger=logger)

    test_cases, elapsed_test = run_inference_records(engine=engine, records=test_records, split_name="test", mode=mode)
    hard_cases, elapsed_hard = run_inference_records(
        engine=engine,
        records=hard_records,
        split_name="hard_test",
        mode=mode,
    )
    inference_cases = test_cases + hard_cases

    hallucination = evaluate_hallucination(
        inference_cases=inference_cases,
        report_json_path=paths["hallucination_report_json"],
        report_md_path=paths["hallucination_report_md"],
    )
    security = evaluate_security_adversarial(
        engine=engine,
        prompt_count=int(mode_cfg["adversarial_prompt_count"]),
        report_json_path=paths["security_adversarial_report_json"],
        report_md_path=paths["security_adversarial_report_md"],
    )
    pii = evaluate_pii_leakage(
        engine=engine,
        inference_cases=inference_cases,
        report_json_path=paths["pii_leak_report_json"],
        report_md_path=paths["pii_leak_report_md"],
    )
    task = evaluate_task_metrics(
        inference_cases=inference_cases,
        taxonomy_path=config["taxonomy"]["taxonomy_file"],
        throughput_window_seconds=(elapsed_test + elapsed_hard),
        report_json_path=paths["task_metrics_report_json"],
        report_md_path=paths["task_metrics_report_md"],
    )
    task = _update_task_report_with_chat_metric(
        task_report=task,
        engine=engine,
        records=test_records,
        max_attempts=int(eval_cfg["chat"]["max_attempts"]),
    )
    write_json(paths["task_metrics_report_json"], task)
    _write_task_markdown(paths["task_metrics_report_md"], task)

    gates = eval_cfg["safety_gates"]
    evidence_mismatch_threshold = float(gates.get("evidence_mismatch_rate_max", 1.0))
    gate_results = {
        "hallucination_rate_actionable": {
            "value": hallucination["metrics"]["hallucination_rate_actionable"],
            "threshold": float(gates["hallucination_rate_actionable_max"]),
            "pass": hallucination["metrics"]["hallucination_rate_actionable"]
            <= float(gates["hallucination_rate_actionable_max"]),
        },
        "evidence_mismatch_rate": {
            "value": hallucination["metrics"]["evidence_mismatch_rate"],
            "threshold": evidence_mismatch_threshold,
            "pass": hallucination["metrics"]["evidence_mismatch_rate"] <= evidence_mismatch_threshold,
        },
        "step_hallucination_rate": {
            "value": hallucination["metrics"]["step_hallucination_rate"],
            "threshold": float(gates["step_hallucination_rate_max"]),
            "pass": hallucination["metrics"]["step_hallucination_rate"] <= float(gates["step_hallucination_rate_max"]),
        },
        "citation_hallucination_rate": {
            "value": hallucination["metrics"]["citation_hallucination_rate"],
            "threshold": float(gates["citation_hallucination_rate_max"]),
            "pass": hallucination["metrics"]["citation_hallucination_rate"]
            <= float(gates["citation_hallucination_rate_max"]),
        },
        "pii_leakage_rate": {
            "value": pii["metrics"]["pii_leakage_rate"],
            "threshold": float(gates["pii_leakage_rate_max"]),
            "pass": pii["metrics"]["pii_leakage_rate"] <= float(gates["pii_leakage_rate_max"]),
        },
        "unsafe_compliance_rate": {
            "value": security["metrics"]["unsafe_compliance_rate"],
            "threshold": float(gates["unsafe_compliance_rate_max"]),
            "pass": security["metrics"]["unsafe_compliance_rate"] <= float(gates["unsafe_compliance_rate_max"]),
        },
    }
    failed = [name for name, payload in gate_results.items() if not payload["pass"]]
    dashboard = {
        "mode": mode,
        "counts": {
            "test_cases": len(test_cases),
            "hard_test_cases": len(hard_cases),
        },
        "key_metrics": {
            "hallucination_rate_actionable": hallucination["metrics"]["hallucination_rate_actionable"],
            "evidence_mismatch_rate": hallucination["metrics"]["evidence_mismatch_rate"],
            "step_hallucination_rate": hallucination["metrics"]["step_hallucination_rate"],
            "citation_hallucination_rate": hallucination["metrics"]["citation_hallucination_rate"],
            "template_compliance_rate": hallucination["metrics"]["template_compliance_rate"],
            "unsafe_compliance_rate": security["metrics"]["unsafe_compliance_rate"],
            "injection_resistance_rate": security["metrics"]["injection_resistance_rate"],
            "pii_leakage_rate": pii["metrics"]["pii_leakage_rate"],
            "intent_macro_f1": task["intent_metrics"]["macro_f1"],
            "step_validity_rate": task["step_quality"]["step_validity_rate"],
            "latency_p95_ms": task["performance"]["latency_p95_ms"],
        },
        "safety_gates": gate_results,
        "overall_pass": len(failed) == 0,
        "failed_gates": failed,
    }
    write_json(paths["combined_dashboard_json"], dashboard)

    if failed:
        gate_key = "hallucination_violation"
        if "pii_leakage_rate" in failed:
            gate_key = "pii_leak"
        handle_gate_violation(
            config=config,
            mode=mode,
            stage="eval",
            gate_key=gate_key,
            reason_code="EVAL_SAFETY_GATE_FAILED",
            message=f"Evaluation safety gates failed: {', '.join(failed)}",
            details={"failed_gates": failed, "dashboard_path": paths["combined_dashboard_json"]},
            logger=logger,
        )

    summary = {
        "hallucination": hallucination["metrics"],
        "security": security["metrics"],
        "pii": pii["metrics"],
        "task": {
            "intent_macro_f1": task["intent_metrics"]["macro_f1"],
            "step_validity_rate": task["step_quality"]["step_validity_rate"],
            "latency_p95_ms": task["performance"]["latency_p95_ms"],
        },
        "dashboard": dashboard,
    }
    log_event(
        logger,
        "INFO",
        "eval_stage_complete",
        {
            "overall_pass": dashboard["overall_pass"],
            "failed_gates": dashboard["failed_gates"],
            "outputs": {
                "hallucination": paths["hallucination_report_json"],
                "security": paths["security_adversarial_report_json"],
                "pii": paths["pii_leak_report_json"],
                "task": paths["task_metrics_report_json"],
                "dashboard": paths["combined_dashboard_json"],
            },
        },
    )
    return summary
