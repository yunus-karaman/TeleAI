from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

from data.schemas import EvidencePack, NormalizedComplaint
from graph.retrieval import RetrievalResources, retrieve_evidence_pack


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _paraphrases(text: str) -> list[str]:
    replacements = [
        ("internet", "baglanti"),
        ("sorun", "problem"),
        ("hiz", "performans"),
        ("kesinti", "kopma"),
    ]
    variants = [text]
    lowered = text.lower()
    for old, new in replacements:
        if old in lowered:
            variants.append(text.replace(old, new))
            variants.append(text.replace(old.capitalize(), new.capitalize()))
            break
    variants.append(f"Lutfen yardimci olun: {text}")
    deduped = []
    seen = set()
    for variant in variants:
        key = variant.strip()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(key)
    return deduped[:3]


def _latency_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"p50_ms": 0.0, "p95_ms": 0.0, "mean_ms": 0.0}
    ordered = sorted(values)
    p50 = ordered[max(0, int(np.ceil(len(ordered) * 0.5)) - 1)]
    p95 = ordered[max(0, int(np.ceil(len(ordered) * 0.95)) - 1)]
    return {"p50_ms": round(float(p50), 3), "p95_ms": round(float(p95), 3), "mean_ms": round(float(mean(values)), 3)}


def _compute_correlation(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 3 or len(xs) != len(ys):
        return 0.0
    arr_x = np.array(xs, dtype=np.float64)
    arr_y = np.array(ys, dtype=np.float64)
    if float(np.std(arr_x)) <= 1e-12 or float(np.std(arr_y)) <= 1e-12:
        return 0.0
    corr = np.corrcoef(arr_x, arr_y)[0, 1]
    if np.isnan(corr):
        return 0.0
    return round(float(corr), 6)


def _resource_snapshot() -> dict[str, Any]:
    gpu_memory_mb = None
    try:
        import torch

        if torch.cuda.is_available():
            gpu_memory_mb = round(float(torch.cuda.max_memory_allocated() / (1024**2)), 3)
    except ModuleNotFoundError:
        gpu_memory_mb = None
    except ImportError:
        gpu_memory_mb = None
    except RuntimeError:
        gpu_memory_mb = None
    return {"gpu_memory_mb": gpu_memory_mb}


def evaluate_retrieval(
    *,
    resources: RetrievalResources,
    test_records: list[NormalizedComplaint],
    hard_test_records: list[NormalizedComplaint],
    review_pack_path: str | Path,
    review_pack_size: int,
    include_debug: bool,
) -> dict[str, Any]:
    weak_predictions: dict[str, list[dict[str, Any]]] = {"test": [], "hard_test": []}
    latencies_test: list[float] = []
    latencies_hard: list[float] = []
    confidences: list[float] = []
    coherences: list[float] = []
    evidence_coverages: list[float] = []
    evidence_redundancy: list[float] = []
    evidence_confidences: list[float] = []
    step_diversity_by_category: dict[str, set[str]] = defaultdict(set)
    step_count_by_category: Counter[str] = Counter()

    def run_split(records: list[NormalizedComplaint], split_name: str, latency_list: list[float]) -> None:
        for record in records:
            pack, telemetry = retrieve_evidence_pack(
                complaint_text=record.complaint_text_clean,
                resources=resources,
                request_id=f"{split_name}:{record.complaint_id}",
                brand=record.brand_slug,
                include_debug=include_debug,
            )
            latency_list.append(telemetry["latency_ms"])
            weak_predictions[split_name].append(
                {
                    "complaint_id": record.complaint_id,
                    "true_category": record.normalized_category,
                    "predicted_category": pack.normalized_category,
                    "category_confidence": pack.category_confidence,
                    "top_step_ids": [item.step_id for item in pack.top_steps],
                    "top_step_scores": [item.step_score for item in pack.top_steps],
                    "evidence_ids": [item.paragraph_id for item in pack.evidence],
                }
            )

            mean_step_score = float(mean([item.step_score for item in pack.top_steps])) if pack.top_steps else 0.0
            confidences.append(pack.category_confidence)
            coherences.append(mean_step_score)

            valid_steps = sum(1 for item in pack.top_steps if len(item.evidence_ids) > 0)
            evidence_coverages.append(valid_steps / max(1, len(pack.top_steps)))

            all_evidence = [evidence_id for item in pack.top_steps for evidence_id in item.evidence_ids]
            redundancy = 1.0 - (len(set(all_evidence)) / max(1, len(all_evidence)))
            evidence_redundancy.append(redundancy)
            evidence_confidences.extend([item.confidence for item in pack.evidence])

            category = record.normalized_category
            step_count_by_category[category] += len(pack.top_steps)
            for step in pack.top_steps:
                step_diversity_by_category[category].add(step.step_id)

    run_split(test_records, "test", latencies_test)
    run_split(hard_test_records, "hard_test", latencies_hard)

    stability_samples = sorted(test_records, key=lambda item: item.complaint_id)[: min(100, len(test_records))]
    stability_scores: list[float] = []
    for record in stability_samples:
        base_pack, _ = retrieve_evidence_pack(
            complaint_text=record.complaint_text_clean,
            resources=resources,
            request_id=f"stability:{record.complaint_id}:base",
            include_debug=False,
        )
        base_steps = {item.step_id for item in base_pack.top_steps}
        for idx, variant in enumerate(_paraphrases(record.complaint_text_clean)[1:], start=1):
            variant_pack, _ = retrieve_evidence_pack(
                complaint_text=variant,
                resources=resources,
                request_id=f"stability:{record.complaint_id}:{idx}",
                include_debug=False,
            )
            variant_steps = {item.step_id for item in variant_pack.top_steps}
            stability_scores.append(_jaccard(base_steps, variant_steps))

    diversity_ratio_by_category = {}
    for category, unique_steps in step_diversity_by_category.items():
        total = step_count_by_category[category]
        diversity_ratio_by_category[category] = round(len(unique_steps) / max(1, total), 6)

    relevance_dist = {
        "mean": round(float(mean(evidence_confidences)), 6) if evidence_confidences else 0.0,
        "p50": round(float(np.percentile(evidence_confidences, 50)), 6) if evidence_confidences else 0.0,
        "p90": round(float(np.percentile(evidence_confidences, 90)), 6) if evidence_confidences else 0.0,
    }

    resource_info = _resource_snapshot()
    cpu_fallback = _latency_stats(latencies_test + latencies_hard)

    report = {
        "weak_label_predictions": {
            "test_count": len(weak_predictions["test"]),
            "hard_test_count": len(weak_predictions["hard_test"]),
        },
        "step_retrieval_metrics": {
            "step_diversity_ratio_by_category": diversity_ratio_by_category,
            "stability_jaccard_mean": round(float(mean(stability_scores)), 6) if stability_scores else 0.0,
            "stability_sample_count": len(stability_scores),
            "confidence_coherence_correlation": _compute_correlation(confidences, coherences),
        },
        "evidence_quality_metrics": {
            "evidence_coverage_mean": round(float(mean(evidence_coverages)), 6) if evidence_coverages else 0.0,
            "evidence_redundancy_mean": round(float(mean(evidence_redundancy)), 6) if evidence_redundancy else 0.0,
            "evidence_relevance_proxy": relevance_dist,
        },
        "latency_resource_metrics": {
            "test": _latency_stats(latencies_test),
            "hard_test": _latency_stats(latencies_hard),
            "cpu_fallback": cpu_fallback,
            **resource_info,
        },
    }

    review_pack = _build_review_pack(
        resources=resources,
        records=test_records,
        sample_size=review_pack_size,
    )
    _write_jsonl(review_pack_path, review_pack)
    return report


def _build_review_pack(
    *,
    resources: RetrievalResources,
    records: list[NormalizedComplaint],
    sample_size: int,
) -> list[dict[str, Any]]:
    per_category: dict[str, list[NormalizedComplaint]] = defaultdict(list)
    for record in records:
        per_category[record.normalized_category].append(record)

    selected: list[NormalizedComplaint] = []
    categories = sorted(per_category.keys())
    if categories:
        quota = max(1, sample_size // len(categories))
        for category in categories:
            candidates = sorted(per_category[category], key=lambda item: item.complaint_id)
            selected.extend(candidates[:quota])
    if len(selected) < sample_size:
        seen_ids = {item.complaint_id for item in selected}
        for record in sorted(records, key=lambda item: item.complaint_id):
            if record.complaint_id in seen_ids:
                continue
            selected.append(record)
            seen_ids.add(record.complaint_id)
            if len(selected) >= sample_size:
                break
    selected = selected[:sample_size]

    review_rows: list[dict[str, Any]] = []
    for record in selected:
        pack, _ = retrieve_evidence_pack(
            complaint_text=record.complaint_text_clean,
            resources=resources,
            request_id=f"review:{record.complaint_id}",
            include_debug=False,
        )
        review_rows.append(
            {
                "complaint_id": record.complaint_id,
                "complaint_text": record.complaint_text_clean[:1000],
                "predicted_category": pack.normalized_category,
                "category_confidence": pack.category_confidence,
                "top_5_steps": [
                    {
                        "step_id": step.step_id,
                        "title_tr": step.title_tr,
                        "level": step.level,
                        "step_score": step.step_score,
                    }
                    for step in pack.top_steps[:5]
                ],
                "evidence": [
                    {
                        "paragraph_id": item.paragraph_id,
                        "text_tr": item.text_tr,
                        "confidence": item.confidence,
                    }
                    for item in pack.evidence[:5]
                ],
            }
        )
    return review_rows


def _write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False))
            out.write("\n")


def write_retrieval_markdown(report: dict[str, Any], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Graph-RAG Retrieval Evaluation",
        "",
        "## Weak Label Coverage",
        f"- test_count: `{report['weak_label_predictions']['test_count']}`",
        f"- hard_test_count: `{report['weak_label_predictions']['hard_test_count']}`",
        "",
        "## Step Retrieval Metrics",
        f"- stability_jaccard_mean: `{report['step_retrieval_metrics']['stability_jaccard_mean']}`",
        f"- confidence_coherence_correlation: `{report['step_retrieval_metrics']['confidence_coherence_correlation']}`",
        "",
        "## Evidence Quality",
        f"- evidence_coverage_mean: `{report['evidence_quality_metrics']['evidence_coverage_mean']}`",
        f"- evidence_redundancy_mean: `{report['evidence_quality_metrics']['evidence_redundancy_mean']}`",
        f"- evidence_relevance_mean: `{report['evidence_quality_metrics']['evidence_relevance_proxy']['mean']}`",
        "",
        "## Latency",
        f"- test p50/p95 (ms): `{report['latency_resource_metrics']['test']['p50_ms']}` / `{report['latency_resource_metrics']['test']['p95_ms']}`",
        f"- hard_test p50/p95 (ms): `{report['latency_resource_metrics']['hard_test']['p50_ms']}` / `{report['latency_resource_metrics']['hard_test']['p95_ms']}`",
        f"- cpu_fallback p50/p95 (ms): `{report['latency_resource_metrics']['cpu_fallback']['p50_ms']}` / `{report['latency_resource_metrics']['cpu_fallback']['p95_ms']}`",
        "",
        "## Step Diversity (Top Categories)",
    ]
    diversity = report["step_retrieval_metrics"]["step_diversity_ratio_by_category"]
    for category, ratio in sorted(diversity.items(), key=lambda item: (-item[1], item[0]))[:15]:
        lines.append(f"- {category}: `{ratio}`")

    path.write_text("\n".join(lines), encoding="utf-8")
