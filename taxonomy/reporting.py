from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from data.schemas import NormalizedComplaint
from taxonomy.baselines import BaselineEvaluation
from taxonomy.schema import TaxonomyFile


def _class_distribution(records: list[NormalizedComplaint]) -> dict[str, int]:
    counts = Counter(record.normalized_category for record in records)
    return {category: int(count) for category, count in sorted(counts.items(), key=lambda item: item[0])}


def _imbalance_ratio(distribution: dict[str, int]) -> float:
    non_zero = [value for value in distribution.values() if value > 0]
    if len(non_zero) < 2:
        return 1.0
    return round(max(non_zero) / max(1, min(non_zero)), 6)


def _sample_needs_review(records: list[NormalizedComplaint], sample_size: int) -> list[dict[str, Any]]:
    candidates = [record for record in records if record.needs_review]
    candidates.sort(key=lambda item: (item.confidence_score, item.complaint_id))
    sampled = candidates[:sample_size]
    return [
        {
            "complaint_id": record.complaint_id,
            "category": record.normalized_category,
            "confidence_score": record.confidence_score,
            "text_snippet": record.complaint_text_clean[:300],
        }
        for record in sampled
    ]


def export_error_analysis(
    *,
    output_dir: str | Path,
    test_records: list[NormalizedComplaint],
    baseline_result: BaselineEvaluation | None,
    assignment_debug: list[dict[str, Any]],
    top_n: int,
) -> dict[str, str]:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)

    mistakes_by_class_path = directory / "mistakes_by_class.json"
    confusion_pairs_path = directory / "confusion_pair_examples.json"
    disagreement_path = directory / "rule_embedding_disagreements.json"

    mistakes_payload: dict[str, list[dict[str, Any]]] = defaultdict(list)
    confusion_payload: dict[str, list[dict[str, Any]]] = defaultdict(list)
    disagreements = [
        item
        for item in assignment_debug
        if item["rule_top_category"] != item["embedding_top_category"]
    ]
    disagreements.sort(key=lambda item: (item["confidence_score"], item["complaint_id"]))

    if baseline_result is not None:
        y_true = [record.normalized_category for record in test_records]
        y_pred = baseline_result.test_predictions
        for record, truth, pred in zip(test_records, y_true, y_pred):
            if truth == pred:
                continue
            mistake_item = {
                "complaint_id": record.complaint_id,
                "actual": truth,
                "predicted": pred,
                "confidence_score": record.confidence_score,
                "text_snippet": record.complaint_text_clean[:260],
            }
            if len(mistakes_payload[truth]) < top_n:
                mistakes_payload[truth].append(mistake_item)
            key = f"{truth}__{pred}"
            if len(confusion_payload[key]) < top_n:
                confusion_payload[key].append(mistake_item)

    mistakes_by_class_path.write_text(
        json.dumps(mistakes_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    confusion_pairs_path.write_text(
        json.dumps(confusion_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    disagreement_path.write_text(
        json.dumps(disagreements[:top_n * 8], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {
        "mistakes_by_class": str(mistakes_by_class_path),
        "confusion_pair_examples": str(confusion_pairs_path),
        "rule_embedding_disagreements": str(disagreement_path),
    }


def build_taxonomy_report(
    *,
    taxonomy: TaxonomyFile,
    labeled_records: list[NormalizedComplaint],
    split_counts: dict[str, int],
    duplicate_cluster_cross_split_count: int,
    baseline_results: dict[str, BaselineEvaluation],
    needs_review_sample_size: int,
    assignment_debug: list[dict[str, Any]],
    test_records: list[NormalizedComplaint],
    error_export_paths: dict[str, str],
) -> dict[str, Any]:
    distribution = _class_distribution(labeled_records)
    low_confidence_rate = round(
        sum(1 for record in labeled_records if record.needs_review) / max(1, len(labeled_records)),
        6,
    )
    imbalance_ratio = _imbalance_ratio(distribution)

    baseline_section: dict[str, Any] = {}
    for name, result in baseline_results.items():
        baseline_section[name] = {
            "test": result.test_metrics,
            "hard_test": result.hard_test_metrics,
        }

    hard_gap: dict[str, Any] = {}
    for name, metrics in baseline_section.items():
        hard_gap[name] = {
            "macro_f1_gap": round(
                float(metrics["test"]["macro_f1"]) - float(metrics["hard_test"]["macro_f1"]),
                6,
            ),
            "accuracy_gap": round(
                float(metrics["test"]["accuracy"]) - float(metrics["hard_test"]["accuracy"]),
                6,
            ),
        }

    category_summary = [
        {
            "category_id": category.category_id,
            "title_tr": category.title_tr,
            "risk_level_default": category.risk_level_default,
        }
        for category in taxonomy.categories
    ]

    rule_embedding_disagreement_rate = round(
        sum(1 for item in assignment_debug if item["rule_top_category"] != item["embedding_top_category"])
        / max(1, len(assignment_debug)),
        6,
    )

    return {
        "taxonomy": {
            "name": taxonomy.taxonomy_name,
            "version": taxonomy.taxonomy_version,
            "category_count": len(taxonomy.categories),
            "categories": category_summary,
        },
        "dataset": {
            "total_records": len(labeled_records),
            "split_counts": split_counts,
            "duplicate_cluster_cross_split_count": duplicate_cluster_cross_split_count,
        },
        "class_distribution": distribution,
        "imbalance_ratio": imbalance_ratio,
        "low_confidence_rate": low_confidence_rate,
        "rule_embedding_disagreement_rate": rule_embedding_disagreement_rate,
        "baselines": baseline_section,
        "hard_test_gap": hard_gap,
        "needs_review_examples": _sample_needs_review(labeled_records, sample_size=needs_review_sample_size),
        "error_analysis_exports": error_export_paths,
    }


def write_taxonomy_markdown(report: dict[str, Any], output_path: str | Path) -> None:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append("# Taxonomy Classification Report")
    lines.append("")
    lines.append(f"- Taxonomy version: `{report['taxonomy']['version']}`")
    lines.append(f"- Category count: `{report['taxonomy']['category_count']}`")
    lines.append(f"- Total labeled records: `{report['dataset']['total_records']}`")
    lines.append(f"- Low confidence rate: `{report['low_confidence_rate']:.4f}`")
    lines.append(f"- Imbalance ratio (max/min): `{report['imbalance_ratio']:.4f}`")
    lines.append("")
    lines.append("## Split Counts")
    for split_name, count in report["dataset"]["split_counts"].items():
        lines.append(f"- {split_name}: `{count}`")
    lines.append("")
    lines.append("## Top Category Distribution")
    for category_id, count in sorted(
        report["class_distribution"].items(),
        key=lambda item: (-item[1], item[0]),
    )[:12]:
        lines.append(f"- {category_id}: `{count}`")
    lines.append("")
    lines.append("## Baseline Metrics")
    for baseline_name, metrics in report["baselines"].items():
        test_metrics = metrics["test"]
        hard_metrics = metrics["hard_test"]
        gap = report["hard_test_gap"][baseline_name]
        lines.append(f"### {baseline_name}")
        lines.append(f"- Test Accuracy: `{test_metrics['accuracy']}`")
        lines.append(f"- Test Macro-F1: `{test_metrics['macro_f1']}`")
        lines.append(f"- Hard Test Accuracy: `{hard_metrics['accuracy']}`")
        lines.append(f"- Hard Test Macro-F1: `{hard_metrics['macro_f1']}`")
        lines.append(f"- Macro-F1 Gap: `{gap['macro_f1_gap']}`")
        top_confusions = test_metrics.get("top_confusions", [])[:5]
        if top_confusions:
            lines.append("- Top confusion pairs:")
            for item in top_confusions:
                lines.append(
                    f"  - {item['actual']} -> {item['predicted']}: {item['count']}"
                )
    lines.append("")
    lines.append("## Needs Review Samples")
    for sample in report["needs_review_examples"][:10]:
        lines.append(
            f"- {sample['complaint_id']} | {sample['category']} | conf={sample['confidence_score']:.4f} | {sample['text_snippet'][:120]}"
        )
    lines.append("")
    lines.append("## Error Analysis Exports")
    for key, value in report["error_analysis_exports"].items():
        lines.append(f"- {key}: `{value}`")

    out.write_text("\n".join(lines), encoding="utf-8")

