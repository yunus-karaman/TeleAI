from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

from data.schemas import GraphEdge, GraphNode, KBParagraph, NormalizedComplaint, SolutionStep
from solution_steps.generator import StepKBLink
from taxonomy.schema import TaxonomyFile


def _edge_id(relation_type: str, source: str, target: str) -> str:
    digest = hashlib.sha256(f"{relation_type}|{source}|{target}".encode("utf-8")).hexdigest()[:16]
    return f"EDGE.{relation_type}.{digest}"


def _time_bucket(created_at_iso: str | None) -> str | None:
    if not created_at_iso:
        return None
    try:
        dt = datetime.fromisoformat(created_at_iso.replace("Z", "+00:00"))
        return f"{dt.year:04d}-{dt.month:02d}"
    except ValueError:
        return None


def _recommend_weight(step: SolutionStep) -> float:
    base = 0.95 if step.level == "L1" else 0.82 if step.level == "L2" else 0.74
    tag_bonus = min(0.12, 0.02 * len(step.tags))
    risk_penalty = 0.05 if step.risk_level == "high" else 0.02 if step.risk_level == "medium" else 0.0
    return round(max(0.1, min(1.0, base + tag_bonus - risk_penalty)), 6)


@dataclass
class GraphBuildResult:
    nodes: list[GraphNode]
    edges: list[GraphEdge]
    stats: dict[str, Any]


def build_graph(
    *,
    complaints: list[NormalizedComplaint],
    taxonomy: TaxonomyFile,
    steps: list[SolutionStep],
    kb_items: list[KBParagraph],
    links: list[StepKBLink],
    include_brand_nodes: bool,
    include_time_bucket_nodes: bool,
) -> GraphBuildResult:
    nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []
    node_ids = set()

    def add_node(node: GraphNode) -> None:
        if node.node_id in node_ids:
            return
        node_ids.add(node.node_id)
        nodes.append(node)

    for category in taxonomy.categories:
        add_node(
            GraphNode(
                node_id=category.category_id,
                node_type="category",
                label=category.title_tr,
                attributes={
                    "title_tr": category.title_tr,
                    "risk_level_default": category.risk_level_default,
                },
                source_ids=[],
                confidence=1.0,
            )
        )

    steps_sorted = sorted(steps, key=lambda item: item.step_id)
    for step in steps_sorted:
        add_node(
            GraphNode(
                node_id=step.step_id,
                node_type="solution_step",
                label=step.title_tr,
                attributes={
                    "category_id": step.category_id,
                    "level": step.level,
                    "risk_level": step.risk_level,
                    "tags_joined": ",".join(step.tags),
                },
                source_ids=[],
                confidence=1.0,
            )
        )
        edges.append(
            GraphEdge(
                edge_id=_edge_id("RECOMMENDS_STEP", step.category_id, step.step_id),
                source_node_id=step.category_id,
                target_node_id=step.step_id,
                relation_type="RECOMMENDS_STEP",
                weight=_recommend_weight(step),
                evidence_ids=[],
                bidirectional=False,
            )
        )

    kb_by_id = {item.paragraph_id: item for item in kb_items}
    for item in sorted(kb_items, key=lambda kb: kb.paragraph_id):
        add_node(
            GraphNode(
                node_id=item.paragraph_id,
                node_type="kb_paragraph",
                label=item.doc_id,
                attributes={
                    "doc_id": item.doc_id,
                    "confidence": item.confidence,
                    "source_type": item.source_type,
                },
                source_ids=item.applies_to_step_ids,
                confidence=item.confidence,
            )
        )

    for link in sorted(links, key=lambda item: item.step_id):
        for evidence_id in link.evidence_ids:
            kb_item = kb_by_id.get(evidence_id)
            if kb_item is None:
                continue
            edges.append(
                GraphEdge(
                    edge_id=_edge_id("SUPPORTED_BY", link.step_id, evidence_id),
                    source_node_id=link.step_id,
                    target_node_id=evidence_id,
                    relation_type="SUPPORTED_BY",
                    weight=round(float(kb_item.confidence), 6),
                    evidence_ids=[evidence_id],
                    bidirectional=False,
                )
            )

    brand_nodes_added: set[str] = set()
    time_bucket_nodes_added: set[str] = set()
    for complaint in sorted(complaints, key=lambda item: item.complaint_id):
        brand_value = complaint.brand_slug or complaint.brand_name or ""
        bucket_value = _time_bucket(complaint.created_at_iso)
        add_node(
            GraphNode(
                node_id=complaint.complaint_id,
                node_type="complaint",
                label=complaint.complaint_id,
                attributes={
                    "brand": brand_value,
                    "time_bucket": bucket_value or "",
                    "normalized_category": complaint.normalized_category,
                    "category_confidence": complaint.confidence_score,
                    "flags_joined": ",".join(complaint.quality_flags),
                    "needs_review": complaint.needs_review,
                },
                source_ids=[],
                confidence=complaint.confidence_score,
            )
        )

        edges.append(
            GraphEdge(
                edge_id=_edge_id("CLASSIFIED_AS", complaint.complaint_id, complaint.normalized_category),
                source_node_id=complaint.complaint_id,
                target_node_id=complaint.normalized_category,
                relation_type="CLASSIFIED_AS",
                weight=round(float(complaint.confidence_score), 6),
                evidence_ids=[],
                bidirectional=False,
            )
        )

        if include_brand_nodes and brand_value:
            brand_node_id = f"brand::{brand_value}"
            if brand_node_id not in brand_nodes_added:
                add_node(
                    GraphNode(
                        node_id=brand_node_id,
                        node_type="brand",
                        label=brand_value,
                        attributes={},
                        source_ids=[],
                        confidence=1.0,
                    )
                )
                brand_nodes_added.add(brand_node_id)
            edges.append(
                GraphEdge(
                    edge_id=_edge_id("HAS_BRAND", complaint.complaint_id, brand_node_id),
                    source_node_id=complaint.complaint_id,
                    target_node_id=brand_node_id,
                    relation_type="HAS_BRAND",
                    weight=1.0,
                    evidence_ids=[],
                    bidirectional=False,
                )
            )

        if include_time_bucket_nodes and bucket_value:
            bucket_node_id = f"time::{bucket_value}"
            if bucket_node_id not in time_bucket_nodes_added:
                add_node(
                    GraphNode(
                        node_id=bucket_node_id,
                        node_type="time_bucket",
                        label=bucket_value,
                        attributes={},
                        source_ids=[],
                        confidence=1.0,
                    )
                )
                time_bucket_nodes_added.add(bucket_node_id)
            edges.append(
                GraphEdge(
                    edge_id=_edge_id("HAS_TIME_BUCKET", complaint.complaint_id, bucket_node_id),
                    source_node_id=complaint.complaint_id,
                    target_node_id=bucket_node_id,
                    relation_type="HAS_TIME_BUCKET",
                    weight=1.0,
                    evidence_ids=[],
                    bidirectional=False,
                )
            )

    nodes = sorted(nodes, key=lambda node: node.node_id)
    edges = sorted(edges, key=lambda edge: edge.edge_id)
    stats = compute_graph_stats(nodes=nodes, edges=edges)
    return GraphBuildResult(nodes=nodes, edges=edges, stats=stats)


def _quantiles(values: list[int], q: float) -> int:
    if not values:
        return 0
    index = max(0, math.ceil(len(values) * q) - 1)
    ordered = sorted(values)
    return int(ordered[index])


def compute_graph_stats(nodes: list[GraphNode], edges: list[GraphEdge]) -> dict[str, Any]:
    node_counts = Counter(node.node_type for node in nodes)
    edge_counts = Counter(edge.relation_type for edge in edges)

    degree = Counter()
    adjacency: dict[str, set[str]] = defaultdict(set)
    for edge in edges:
        degree[edge.source_node_id] += 1
        degree[edge.target_node_id] += 1
        adjacency[edge.source_node_id].add(edge.target_node_id)
        adjacency[edge.target_node_id].add(edge.source_node_id)

    isolated = [node.node_id for node in nodes if degree[node.node_id] == 0]
    degree_values = [degree[node.node_id] for node in nodes]
    degree_by_type: dict[str, dict[str, float]] = {}
    for node_type in node_counts:
        values = [degree[node.node_id] for node in nodes if node.node_type == node_type]
        degree_by_type[node_type] = {
            "count": len(values),
            "min": min(values) if values else 0,
            "max": max(values) if values else 0,
            "mean": round(mean(values), 6) if values else 0.0,
            "p50": _quantiles(values, 0.5),
            "p95": _quantiles(values, 0.95),
        }

    visited = set()
    component_sizes: list[int] = []
    for node in nodes:
        node_id = node.node_id
        if node_id in visited:
            continue
        stack = [node_id]
        visited.add(node_id)
        size = 0
        while stack:
            cur = stack.pop()
            size += 1
            for nxt in adjacency.get(cur, set()):
                if nxt not in visited:
                    visited.add(nxt)
                    stack.append(nxt)
        component_sizes.append(size)

    return {
        "node_counts_by_type": dict(node_counts),
        "edge_counts_by_type": dict(edge_counts),
        "isolated_nodes_count": len(isolated),
        "isolated_nodes_sample": isolated[:20],
        "degree_distribution": {
            "count": len(degree_values),
            "min": min(degree_values) if degree_values else 0,
            "max": max(degree_values) if degree_values else 0,
            "mean": round(mean(degree_values), 6) if degree_values else 0.0,
            "p50": _quantiles(degree_values, 0.5),
            "p95": _quantiles(degree_values, 0.95),
        },
        "degree_by_node_type": degree_by_type,
        "connectivity_checks": {
            "weakly_connected_components": len(component_sizes),
            "largest_component_size": max(component_sizes) if component_sizes else 0,
            "all_nodes_covered": sum(component_sizes) == len(nodes),
        },
    }


def write_graph_artifacts(result: GraphBuildResult, nodes_path: Path, edges_path: Path, stats_path: Path) -> None:
    nodes_path.parent.mkdir(parents=True, exist_ok=True)
    with nodes_path.open("w", encoding="utf-8") as out:
        for node in result.nodes:
            out.write(json.dumps(node.model_dump(mode="json"), ensure_ascii=False))
            out.write("\n")

    edges_path.parent.mkdir(parents=True, exist_ok=True)
    with edges_path.open("w", encoding="utf-8") as out:
        for edge in result.edges:
            out.write(json.dumps(edge.model_dump(mode="json"), ensure_ascii=False))
            out.write("\n")

    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(result.stats, ensure_ascii=False, indent=2), encoding="utf-8")

