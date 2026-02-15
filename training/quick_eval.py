from __future__ import annotations

import json
import logging
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

from models.infer import ConstrainedInferenceEngine, load_normalized_records
from scripts.logging_utils import log_event


def _latency_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"p50_ms": 0.0, "p95_ms": 0.0, "mean_ms": 0.0}
    ordered = sorted(values)
    p50 = ordered[max(0, int(np.ceil(len(ordered) * 0.5)) - 1)]
    p95 = ordered[max(0, int(np.ceil(len(ordered) * 0.95)) - 1)]
    return {"p50_ms": round(float(p50), 3), "p95_ms": round(float(p95), 3), "mean_ms": round(float(mean(values)), 3)}


def _gpu_memory_mb() -> float | None:
    try:
        import torch

        if torch.cuda.is_available():
            return round(float(torch.cuda.max_memory_allocated() / (1024**2)), 3)
    except ModuleNotFoundError:
        return None
    except ImportError:
        return None
    except RuntimeError:
        return None
    return None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Training Quick Evaluation",
        "",
        f"- template_compliance_rate: `{payload['metrics']['template_compliance_rate']}`",
        f"- step_validity_rate: `{payload['metrics']['step_validity_rate']}`",
        f"- evidence_citation_coverage: `{payload['metrics']['evidence_citation_coverage']}`",
        f"- pii_detection_rate: `{payload['metrics']['pii_detection_rate']}`",
        f"- latency_p50_ms: `{payload['latency']['p50_ms']}`",
        f"- latency_p95_ms: `{payload['latency']['p95_ms']}`",
        f"- gpu_memory_mb: `{payload['resource']['gpu_memory_mb']}`",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run_training_quick_eval(
    *,
    config: dict[str, Any],
    mode: str,
    logger: logging.Logger,
    run_id: str | None = None,
) -> dict[str, Any]:
    paths = config["paths"]
    eval_cfg = config["training_llm"]["quick_eval"]["SMOKE" if mode == "SMOKE" else "FULL"]

    test_records = load_normalized_records(Path(paths["test_split"]))
    hard_test_records = load_normalized_records(Path(paths["hard_test_split"]))

    if eval_cfg.get("test_limit") is not None:
        test_records = test_records[: int(eval_cfg["test_limit"])]
    if eval_cfg.get("hard_test_limit") is not None:
        hard_test_records = hard_test_records[: int(eval_cfg["hard_test_limit"])]

    eval_records = [("test", record) for record in test_records] + [("hard_test", record) for record in hard_test_records]
    engine = ConstrainedInferenceEngine(config=config, mode=mode, logger=logger, run_id=run_id)

    template_flags: list[float] = []
    step_valid_flags: list[float] = []
    evidence_covs: list[float] = []
    pii_flags: list[float] = []
    latencies: list[float] = []
    failures = 0

    for split_name, record in eval_records:
        try:
            result = engine.infer(record.complaint_text_clean, brand=record.brand_slug)
            validation = result["validation"]
            template_flags.append(1.0 if validation["template_compliant"] else 0.0)
            step_valid_flags.append(1.0 if validation["step_valid"] else 0.0)
            evidence_covs.append(float(validation["evidence_coverage"]))
            pii_flags.append(0.0 if validation["pii_free"] else 1.0)
            latencies.append(float(result["latency_ms"]))
        except Exception as error:
            failures += 1
            log_event(
                logger,
                "WARNING",
                "train_llm_quick_eval_failure",
                {"split": split_name, "complaint_id": record.complaint_id, "error": str(error)},
            )

    total = len(eval_records)
    metrics = {
        "template_compliance_rate": round(float(mean(template_flags)) if template_flags else 0.0, 6),
        "step_validity_rate": round(float(mean(step_valid_flags)) if step_valid_flags else 0.0, 6),
        "evidence_citation_coverage": round(float(mean(evidence_covs)) if evidence_covs else 0.0, 6),
        "pii_detection_rate": round(float(mean(pii_flags)) if pii_flags else 0.0, 6),
    }
    payload = {
        "mode": mode,
        "counts": {
            "test": len(test_records),
            "hard_test": len(hard_test_records),
            "total_evaluated": total,
            "failures": failures,
        },
        "metrics": metrics,
        "latency": _latency_stats(latencies),
        "resource": {"gpu_memory_mb": _gpu_memory_mb()},
        "run_id": run_id,
    }

    _write_json(Path(paths["training_eval_quick_json"]), payload)
    _write_markdown(Path(paths["training_eval_quick_md"]), payload)
    log_event(logger, "INFO", "train_llm_quick_eval_complete", payload["counts"])
    return payload
