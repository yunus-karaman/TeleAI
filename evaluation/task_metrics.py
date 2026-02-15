from __future__ import annotations

from collections import Counter, defaultdict
from statistics import mean
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

from evaluation.common import percentile, write_json, write_markdown
from graph.embeddings import HashingTextEmbedder
from taxonomy.schema import load_taxonomy_file


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


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom <= 0.0:
        return 0.0
    return float(np.dot(left, right) / denom)


def evaluate_task_metrics(
    *,
    inference_cases: list[dict[str, Any]],
    taxonomy_path: str,
    throughput_window_seconds: float,
    report_json_path: str,
    report_md_path: str,
) -> dict[str, Any]:
    y_true: list[str] = []
    y_pred: list[str] = []
    step_valid_flags: list[float] = []
    step_count_flags: list[float] = []
    latencies: list[float] = []
    escalation_by_category: dict[str, list[float]] = defaultdict(list)
    step_diversity: dict[str, list[str]] = defaultdict(list)
    relevance_scores: list[float] = []

    embedder = HashingTextEmbedder(dimension=256, ngram_min=1, ngram_max=2)

    for case in inference_cases:
        inference = case["inference"]
        pack = inference["evidence_pack"]
        validation = inference["validation"]
        category_true = case["true_category"]
        category_pred = pack["normalized_category"]
        y_true.append(category_true)
        y_pred.append(category_pred)

        step_valid_flags.append(1.0 if validation.get("step_valid", False) else 0.0)
        step_count = len(pack["top_steps"])
        step_count_flags.append(1.0 if 3 <= step_count <= 5 else 0.0)
        latencies.append(float(inference.get("latency_ms", 0.0)))

        escalation_flag = 0.0
        esc = pack.get("escalation_suggestion", {})
        if esc.get("unit") and esc.get("unit") != "GENERAL_SUPPORT":
            escalation_flag = 1.0
        if esc.get("threshold_signals"):
            escalation_flag = 1.0
        escalation_by_category[category_pred].append(escalation_flag)

        step_ids = [item["step_id"] for item in pack["top_steps"]]
        step_diversity[category_pred].extend(step_ids)

        complaint_vec = embedder.embed([case["complaint_text_clean"]])[0]
        for step in pack["top_steps"]:
            step_vec = embedder.embed([f"{step['title_tr']} {' '.join(step['instructions_tr'])}"])[0]
            relevance_scores.append(_cosine(complaint_vec, step_vec))

    labels = sorted(set(y_true) | set(y_pred))
    accuracy = float(accuracy_score(y_true, y_pred)) if y_true else 0.0
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)) if y_true else 0.0
    per_class_prf = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    per_class = {
        label: {
            "precision": round(float(per_class_prf[0][idx]), 6),
            "recall": round(float(per_class_prf[1][idx]), 6),
            "f1": round(float(per_class_prf[2][idx]), 6),
            "support": int(per_class_prf[3][idx]),
        }
        for idx, label in enumerate(labels)
    }
    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist() if y_true else []

    diversity_ratio = {
        category: round(len(set(step_ids)) / float(max(1, len(step_ids))), 6)
        for category, step_ids in step_diversity.items()
    }
    escalation_rate_per_category = {
        category: round(float(mean(flags)) if flags else 0.0, 6)
        for category, flags in escalation_by_category.items()
    }

    taxonomy = load_taxonomy_file(taxonomy_path)
    risk_by_category = {item.category_id: item.risk_level_default for item in taxonomy.categories}
    high_risk_rates = [
        escalation_rate_per_category.get(category, 0.0)
        for category, risk in risk_by_category.items()
        if risk == "high"
    ]
    non_high_rates = [
        escalation_rate_per_category.get(category, 0.0)
        for category, risk in risk_by_category.items()
        if risk != "high"
    ]
    high_risk_mean = float(mean(high_risk_rates)) if high_risk_rates else 0.0
    non_high_mean = float(mean(non_high_rates)) if non_high_rates else 0.0

    throughput_rps = (len(inference_cases) / throughput_window_seconds) if throughput_window_seconds > 0 else 0.0
    report = {
        "intent_metrics": {
            "accuracy": round(accuracy, 6),
            "macro_f1": round(macro_f1, 6),
            "per_class": per_class,
            "labels": labels,
            "confusion_matrix": cm,
        },
        "step_quality": {
            "step_validity_rate": round(float(mean(step_valid_flags)) if step_valid_flags else 0.0, 6),
            "step_count_correctness_rate": round(float(mean(step_count_flags)) if step_count_flags else 0.0, 6),
            "step_diversity_per_category": diversity_ratio,
            "proxy_relevance_mean": round(float(mean(relevance_scores)) if relevance_scores else 0.0, 6),
        },
        "escalation": {
            "escalation_rate_per_category": escalation_rate_per_category,
            "high_risk_escalation_mean": round(high_risk_mean, 6),
            "non_high_risk_escalation_mean": round(non_high_mean, 6),
            "high_risk_consistency": bool(high_risk_mean >= non_high_mean),
        },
        "performance": {
            "latency_p50_ms": round(percentile(latencies, 50), 3),
            "latency_p95_ms": round(percentile(latencies, 95), 3),
            "latency_mean_ms": round(float(mean(latencies)) if latencies else 0.0, 3),
            "throughput_rps": round(float(throughput_rps), 6),
            "gpu_memory_mb": _gpu_memory_mb(),
        },
    }

    write_json(report_json_path, report)
    lines = [
        "# Task Metrics Report",
        "",
        f"- intent_accuracy: `{report['intent_metrics']['accuracy']}`",
        f"- intent_macro_f1: `{report['intent_metrics']['macro_f1']}`",
        f"- step_validity_rate: `{report['step_quality']['step_validity_rate']}`",
        f"- step_count_correctness_rate: `{report['step_quality']['step_count_correctness_rate']}`",
        f"- proxy_relevance_mean: `{report['step_quality']['proxy_relevance_mean']}`",
        f"- high_risk_consistency: `{report['escalation']['high_risk_consistency']}`",
        f"- latency_p50_ms: `{report['performance']['latency_p50_ms']}`",
        f"- latency_p95_ms: `{report['performance']['latency_p95_ms']}`",
        f"- throughput_rps: `{report['performance']['throughput_rps']}`",
    ]
    write_markdown(report_md_path, lines)
    return report
