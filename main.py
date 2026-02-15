from __future__ import annotations

import argparse
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from data.ingestion import load_and_validate_raw_complaints
from data.schema_analysis import analyze_dataset_schema
from data.schemas import EvaluationReport
from preprocess.pipeline import run_preprocess_stage
from scripts.config_loader import load_config
from scripts.logging_utils import configure_json_logging, log_event
from scripts.reproducibility import set_global_determinism
from scripts.runtime_gates import write_aborted_reason
from evaluation.pipeline import run_eval_stage
from graph.pipeline import run_graph_stage
from models.infer import run_infer_stage
from solution_steps.pipeline import run_solution_steps_stage
from taxonomy.pipeline import run_taxonomy_stage
from training.pipeline import run_train_llm_stage


def _run_stage_stub(stage_name: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {"stage": stage_name, "status": "stubbed", "payload": payload}


def run_pipeline(
    mode: str,
    config_path: str,
    stage: str = "all",
    input_text: str | None = None,
    run_id: str | None = None,
) -> int:
    config = load_config(config_path=config_path, mode=mode)
    logger = configure_json_logging(
        level=config["logging"]["level"],
        log_file=config["logging"]["file"],
    )

    seed = int(config["reproducibility"]["seed"])
    deterministic = bool(config["reproducibility"]["deterministic"])
    set_global_determinism(seed=seed, deterministic=deterministic)

    artifacts_dir = Path(config["paths"]["artifacts_dir"])
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    log_event(
        logger,
        "INFO",
        "pipeline_start",
        {"mode": mode, "stage": stage, "seed": seed, "deterministic": deterministic},
    )

    dataset_path = config["paths"]["dataset"]
    schema_report_path = config["paths"]["schema_report"]
    quarantine_path = config["paths"]["quarantine"]

    schema_report = analyze_dataset_schema(dataset_path=dataset_path, output_path=schema_report_path)
    log_event(
        logger,
        "INFO",
        "schema_analysis_complete",
        {
            "total_records": schema_report["total_records"],
            "required_fields": schema_report["required_fields"],
            "avg_text_length": schema_report["avg_text_length"],
            "p95_length": schema_report["95th_percentile_length"],
        },
    )

    if stage == "preprocess":
        run_preprocess_stage(config=config, mode=mode, logger=logger)
        return 0
    if stage == "taxonomy":
        run_taxonomy_stage(config=config, mode=mode, logger=logger)
        return 0
    if stage == "solution_steps":
        run_solution_steps_stage(config=config, mode=mode, logger=logger)
        return 0
    if stage == "graph":
        run_graph_stage(config=config, mode=mode, logger=logger)
        return 0
    if stage == "train_llm":
        run_train_llm_stage(config=config, mode=mode, logger=logger)
        return 0
    if stage == "infer":
        if not input_text or not input_text.strip():
            raise ValueError("--input is required for infer stage")
        infer_result = run_infer_stage(
            config=config,
            mode=mode,
            logger=logger,
            complaint_text=input_text.strip(),
            run_id=run_id,
        )
        print(json.dumps(infer_result, ensure_ascii=False, indent=2))
        return 0
    if stage == "eval":
        run_eval_stage(config=config, mode=mode, logger=logger)
        return 0

    if stage != "all":
        raise ValueError(f"Unsupported stage: {stage}")

    preprocess_report = run_preprocess_stage(config=config, mode=mode, logger=logger)
    taxonomy_report = run_taxonomy_stage(config=config, mode=mode, logger=logger)
    solution_summary = run_solution_steps_stage(config=config, mode=mode, logger=logger)
    graph_summary = run_graph_stage(config=config, mode=mode, logger=logger)

    sample_size = config["mode_runtime"]["sample_size"]
    valid_records, ingestion_stats = load_and_validate_raw_complaints(
        dataset_path=dataset_path,
        quarantine_path=quarantine_path,
        sample_size=sample_size,
    )
    log_event(logger, "INFO", "ingestion_complete", ingestion_stats)

    # Downstream stages are intentionally stubs in this foundation phase.
    stage_payload = {"record_count": len(valid_records), "mode": mode}
    for stage_name in ["training", "evaluation"]:
        stage_result = _run_stage_stub(stage_name=stage_name, payload=stage_payload)
        log_event(logger, "INFO", f"stage_{stage_name}_complete", stage_result)

    report = EvaluationReport(
        run_id=str(uuid.uuid4()),
        mode=mode,
        dataset_size=ingestion_stats["total_lines"],
        valid_records=ingestion_stats["valid_records"],
        quarantined_records=ingestion_stats["quarantined_records"],
        hallucination_rate=0.0,
        evidence_coverage=0.0,
        escalation_rate=0.0,
        latency_p95_ms=0.0,
        pass_fail="PASS",
        notes=[
            "Foundation pipeline completed.",
            "Preprocess stage implemented; downstream stages remain stubs.",
        ],
        metrics={
            "schema_total_fields": float(schema_report["total_fields"]),
            "schema_valid_json_records": float(schema_report["total_records"]),
            "preprocess_valid_records": float(preprocess_report["valid_records"]),
            "taxonomy_labeled_records": float(taxonomy_report["dataset"]["total_records"]),
            "solution_steps_count": float(sum(solution_summary["count_per_category"].values())),
            "graph_nodes_count": float(graph_summary["graph"]["nodes"]),
        },
        generated_at_iso=datetime.now(timezone.utc).isoformat(),
    )
    report_path = artifacts_dir / "evaluation_report.json"
    report_path.write_text(json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2), encoding="utf-8")

    log_event(logger, "INFO", "pipeline_complete", {"report_path": str(report_path)})
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Telecom Complaint Assistant Pipeline")
    parser.add_argument(
        "--stage",
        choices=["all", "preprocess", "taxonomy", "solution_steps", "graph", "train_llm", "infer", "eval"],
        default="all",
        help="Pipeline stage to execute",
    )
    parser.add_argument("--mode", choices=["SMOKE", "FULL"], required=True, help="Pipeline execution mode")
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML file")
    parser.add_argument("--input", default=None, help="Input complaint text for infer stage")
    parser.add_argument("--run-id", default=None, help="Optional trained run id for infer stage")
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    try:
        return run_pipeline(
            mode=args.mode,
            config_path=args.config,
            stage=args.stage,
            input_text=args.input,
            run_id=args.run_id,
        )
    except Exception as error:
        try:
            cfg = load_config(config_path=args.config, mode=args.mode)
            write_aborted_reason(
                cfg,
                stage=args.stage,
                reason_code="PIPELINE_EXCEPTION",
                message=str(error),
                details={"exception_type": type(error).__name__},
            )
        except Exception as abort_error:
            print(f"Failed to write aborted_reason.json: {abort_error}", file=sys.stderr)
        print(f"Pipeline failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
