from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import ValidationError

from data.schemas import GraphEdge, GraphNode, KBParagraph, NormalizedComplaint, SolutionStep
from graph.builder import GraphBuildResult, build_graph, write_graph_artifacts
from graph.embeddings import EmbeddingCache, HashingTextEmbedder
from graph.evaluation import evaluate_retrieval, write_retrieval_markdown
from graph.gnn import run_lightweight_gnn, save_gnn_result
from graph.retrieval import build_retrieval_resources
from scripts.logging_utils import log_event
from scripts.solution_dataset_integrity import run_solution_dataset_integrity
from solution_steps.generator import StepKBLink
from taxonomy.schema import TaxonomyFile, load_taxonomy_file


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Required input file not found: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as infile:
        for line in infile:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _load_models(path: Path, model_type: Any) -> list[Any]:
    rows = _load_jsonl(path)
    models = []
    for row in rows:
        try:
            models.append(model_type.model_validate(row))
        except ValidationError as error:
            raise RuntimeError(f"Invalid record in {path}: {error}") from error
    return models


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _vector_for_node(node: GraphNode, text_vectors: dict[str, np.ndarray], embedder: HashingTextEmbedder) -> np.ndarray:
    if node.node_id in text_vectors:
        return text_vectors[node.node_id]
    text = f"{node.label} {node.attributes.get('title_tr','')} {node.attributes.get('tags_joined','')}"
    return embedder.embed([text])[0]


def _build_text_payloads(
    complaints: list[NormalizedComplaint],
    steps: list[SolutionStep],
    kb_items: list[KBParagraph],
    taxonomy: TaxonomyFile,
) -> dict[str, tuple[list[str], list[str]]]:
    complaint_ids = [record.complaint_id for record in complaints]
    complaint_texts = [record.complaint_text_clean for record in complaints]

    steps_sorted = sorted(steps, key=lambda item: item.step_id)
    step_ids = [item.step_id for item in steps_sorted]
    step_texts = [
        f"{item.title_tr} {' '.join(item.instructions_tr)} {' '.join(item.tags)} {item.level} {item.risk_level}"
        for item in steps_sorted
    ]

    kb_sorted = sorted(kb_items, key=lambda item: item.paragraph_id)
    kb_ids = [item.paragraph_id for item in kb_sorted]
    kb_texts = [item.text_tr for item in kb_sorted]

    category_sorted = sorted(taxonomy.categories, key=lambda item: item.category_id)
    category_ids = [item.category_id for item in category_sorted]
    category_texts = [f"{item.title_tr} {item.description_tr} {' '.join(item.keywords_tr)}" for item in category_sorted]

    return {
        "complaints": (complaint_ids, complaint_texts),
        "steps": (step_ids, step_texts),
        "kb": (kb_ids, kb_texts),
        "categories": (category_ids, category_texts),
    }


def run_graph_stage(*, config: dict[str, Any], mode: str, logger: logging.Logger) -> dict[str, Any]:
    run_solution_dataset_integrity(config=config, mode=mode, logger=logger, stage="graph_precheck")

    paths = config["paths"]
    graph_cfg = config["graph_pipeline"]
    graph_mode_cfg = graph_cfg["mode"]["SMOKE" if mode == "SMOKE" else "FULL"]

    taxonomy = load_taxonomy_file(config["taxonomy"]["taxonomy_file"])
    all_labeled = _load_models(Path(paths["labeled_complaints"]), NormalizedComplaint)
    train_records = _load_models(Path(paths["train_split"]), NormalizedComplaint)
    val_records = _load_models(Path(paths["val_split"]), NormalizedComplaint)
    test_records = _load_models(Path(paths["test_split"]), NormalizedComplaint)
    hard_test_records = _load_models(Path(paths["hard_test_split"]), NormalizedComplaint)
    steps = _load_models(Path(paths["solution_steps_jsonl"]), SolutionStep)
    kb_items = _load_models(Path(paths["kb_jsonl"]), KBParagraph)
    links = _load_models(Path(paths["step_kb_links_jsonl"]), StepKBLink)

    complaints_sorted = sorted(all_labeled, key=lambda item: item.complaint_id)
    if graph_mode_cfg["complaint_limit"] is not None:
        complaints_sorted = complaints_sorted[: int(graph_mode_cfg["complaint_limit"])]

    eval_test_records = sorted(test_records, key=lambda item: item.complaint_id)
    eval_hard_records = sorted(hard_test_records, key=lambda item: item.complaint_id)
    if graph_mode_cfg["eval_limit"] is not None:
        eval_limit = int(graph_mode_cfg["eval_limit"])
        eval_test_records = eval_test_records[:eval_limit]
        eval_hard_records = eval_hard_records[: min(eval_limit, len(eval_hard_records))]

    log_event(
        logger,
        "INFO",
        "graph_inputs_loaded",
        {
            "complaints_total": len(all_labeled),
            "complaints_used": len(complaints_sorted),
            "train": len(train_records),
            "val": len(val_records),
            "test_eval": len(eval_test_records),
            "hard_eval": len(eval_hard_records),
            "steps": len(steps),
            "kb": len(kb_items),
            "links": len(links),
        },
    )

    graph_result = build_graph(
        complaints=complaints_sorted,
        taxonomy=taxonomy,
        steps=steps,
        kb_items=kb_items,
        links=links,
        include_brand_nodes=bool(graph_cfg["include_brand_nodes"]),
        include_time_bucket_nodes=bool(graph_cfg["include_time_bucket_nodes"]),
    )

    nodes_path = Path(paths["graph_nodes"])
    edges_path = Path(paths["graph_edges"])
    stats_path = Path(paths["graph_stats"])
    write_graph_artifacts(graph_result, nodes_path=nodes_path, edges_path=edges_path, stats_path=stats_path)

    embedder = HashingTextEmbedder(
        dimension=int(graph_cfg["embeddings"]["dimension"]),
        ngram_min=int(graph_cfg["embeddings"]["ngram_min"]),
        ngram_max=int(graph_cfg["embeddings"]["ngram_max"]),
    )
    cache = EmbeddingCache(paths["embeddings_dir"])
    text_payloads = _build_text_payloads(
        complaints=complaints_sorted,
        steps=steps,
        kb_items=kb_items,
        taxonomy=taxonomy,
    )

    embeddings: dict[str, tuple[list[str], np.ndarray]] = {}
    for key, (ids, texts) in text_payloads.items():
        vectors = cache.get_or_compute(
            key=key,
            ids=ids,
            texts=texts,
            embedder=embedder,
            force_recompute=bool(graph_cfg["embeddings"]["force_recompute"]),
        )
        embeddings[key] = (ids, vectors)

    step_prior_weight = {
        edge.target_node_id: float(edge.weight)
        for edge in graph_result.edges
        if edge.relation_type == "RECOMMENDS_STEP"
    }

    step_gnn_embeddings = None
    category_gnn_embeddings = None
    gnn_artifact_path = None
    if bool(config["graph"].get("use_gnn", False)):
        id_to_vector = {
            node_id: vector for node_id, vector in zip(embeddings["complaints"][0], embeddings["complaints"][1])
        }
        id_to_vector.update({node_id: vector for node_id, vector in zip(embeddings["steps"][0], embeddings["steps"][1])})
        id_to_vector.update({node_id: vector for node_id, vector in zip(embeddings["kb"][0], embeddings["kb"][1])})
        id_to_vector.update(
            {node_id: vector for node_id, vector in zip(embeddings["categories"][0], embeddings["categories"][1])}
        )

        graph_node_ids = [node.node_id for node in graph_result.nodes]
        graph_vectors = np.stack([_vector_for_node(node, id_to_vector, embedder) for node in graph_result.nodes], axis=0)
        gnn_result = run_lightweight_gnn(
            node_ids=graph_node_ids,
            base_embeddings=graph_vectors,
            edges=graph_result.edges,
            epochs=int(graph_cfg["gnn"]["epochs"]),
            self_weight=float(graph_cfg["gnn"]["self_weight"]),
            neighbor_weight=float(graph_cfg["gnn"]["neighbor_weight"]),
            convergence_tol=float(graph_cfg["gnn"]["convergence_tol"]),
        )
        gnn_artifact_path = Path(paths["gnn_embeddings"])
        save_gnn_result(gnn_result, gnn_artifact_path)

        emb_by_node = {node_id: gnn_result.embeddings[idx] for idx, node_id in enumerate(gnn_result.node_ids)}
        step_gnn_embeddings = {step.step_id: emb_by_node[step.step_id] for step in steps if step.step_id in emb_by_node}
        category_gnn_embeddings = {
            category.category_id: emb_by_node[category.category_id]
            for category in taxonomy.categories
            if category.category_id in emb_by_node
        }

    retrieval_resources = build_retrieval_resources(
        taxonomy=taxonomy,
        complaints=complaints_sorted,
        steps=sorted(steps, key=lambda item: item.step_id),
        kb_items=sorted(kb_items, key=lambda item: item.paragraph_id),
        links=sorted(links, key=lambda item: item.step_id),
        embeddings=embeddings,
        embedder_callable=embedder.embed,
        config={
            "taxonomy_assignment": config["taxonomy"]["assignment"],
            "seed": int(config["reproducibility"]["seed"]),
            "alpha": float(graph_cfg["retrieval"]["alpha"]),
            "beta": float(graph_cfg["retrieval"]["beta"]),
            "gamma": float(graph_cfg["retrieval"]["gamma"]),
            "delta": float(graph_cfg["retrieval"]["delta"]),
            "lambda_gnn": float(graph_cfg["retrieval"]["lambda_gnn"]),
            "top_steps": int(graph_cfg["retrieval"]["top_steps"]),
            "min_steps": int(graph_cfg["retrieval"]["min_steps"]),
            "max_evidence": int(graph_cfg["retrieval"]["max_evidence"]),
            "escalation_threshold": float(graph_cfg["retrieval"]["escalation_threshold"]),
        },
        step_prior_weight=step_prior_weight,
        step_gnn_embeddings=step_gnn_embeddings,
        category_gnn_embeddings=category_gnn_embeddings,
    )

    retrieval_eval = evaluate_retrieval(
        resources=retrieval_resources,
        test_records=eval_test_records,
        hard_test_records=eval_hard_records,
        review_pack_path=paths["review_pack_for_humans"],
        review_pack_size=int(graph_cfg["evaluation"]["review_pack_size"]),
        include_debug=bool(graph_mode_cfg["include_retrieval_debug"]),
    )
    retrieval_eval["graph_stats"] = graph_result.stats
    retrieval_eval["use_gnn"] = bool(config["graph"].get("use_gnn", False))
    if gnn_artifact_path is not None:
        retrieval_eval["gnn_embeddings_path"] = str(gnn_artifact_path)

    retrieval_eval_path = Path(paths["retrieval_eval_json"])
    _write_json(retrieval_eval_path, retrieval_eval)
    write_retrieval_markdown(retrieval_eval, output_path=paths["retrieval_eval_md"])

    log_event(
        logger,
        "INFO",
        "graph_stage_complete",
        {
            "graph_nodes": len(graph_result.nodes),
            "graph_edges": len(graph_result.edges),
            "use_gnn": bool(config["graph"].get("use_gnn", False)),
            "outputs": {
                "nodes": paths["graph_nodes"],
                "edges": paths["graph_edges"],
                "stats": paths["graph_stats"],
                "retrieval_eval_json": paths["retrieval_eval_json"],
                "retrieval_eval_md": paths["retrieval_eval_md"],
                "review_pack": paths["review_pack_for_humans"],
            },
        },
    )

    return {
        "graph": {"nodes": len(graph_result.nodes), "edges": len(graph_result.edges)},
        "retrieval_eval": retrieval_eval,
    }
