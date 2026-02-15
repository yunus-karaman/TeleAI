from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Callable

import numpy as np
from pydantic import ValidationError

from data.schemas import (
    ChatMessage,
    KBParagraph,
    NormalizedComplaint,
    SolutionStep,
    Task1IntentExample,
    Task2SFTExample,
)
from graph.embeddings import EmbeddingCache, HashingTextEmbedder
from graph.retrieval import RetrievalResources, build_retrieval_resources, retrieve_evidence_pack
from models.template_renderer import build_strict_system_prompt_tr, render_deterministic_response
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
    records: list[Any] = []
    for row in rows:
        try:
            records.append(model_type.model_validate(row))
        except ValidationError as error:
            raise RuntimeError(f"Invalid record in {path}: {error}") from error
    return records


def _write_jsonl(path: Path, rows: list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as out:
        for row in rows:
            payload = row.model_dump(mode="json") if hasattr(row, "model_dump") else row
            out.write(json.dumps(payload, ensure_ascii=False))
            out.write("\n")


def _build_text_payloads(
    complaints: list[NormalizedComplaint],
    steps: list[SolutionStep],
    kb_items: list[KBParagraph],
    taxonomy: TaxonomyFile,
) -> dict[str, tuple[list[str], list[str]]]:
    complaint_ids = [item.complaint_id for item in complaints]
    complaint_texts = [item.complaint_text_clean for item in complaints]

    steps_sorted = sorted(steps, key=lambda item: item.step_id)
    step_ids = [item.step_id for item in steps_sorted]
    step_texts = [f"{item.title_tr} {' '.join(item.instructions_tr)} {' '.join(item.tags)}" for item in steps_sorted]

    kb_sorted = sorted(kb_items, key=lambda item: item.paragraph_id)
    kb_ids = [item.paragraph_id for item in kb_sorted]
    kb_texts = [item.text_tr for item in kb_sorted]

    cat_sorted = sorted(taxonomy.categories, key=lambda item: item.category_id)
    category_ids = [item.category_id for item in cat_sorted]
    category_texts = [f"{item.title_tr} {item.description_tr} {' '.join(item.keywords_tr)}" for item in cat_sorted]

    return {
        "complaints": (complaint_ids, complaint_texts),
        "steps": (step_ids, step_texts),
        "kb": (kb_ids, kb_texts),
        "categories": (category_ids, category_texts),
    }


def build_retrieval_resources_from_artifacts(
    *,
    config: dict[str, Any],
    mode: str,
    logger: logging.Logger,
) -> tuple[RetrievalResources, list[NormalizedComplaint], list[NormalizedComplaint], list[NormalizedComplaint], list[NormalizedComplaint]]:
    run_solution_dataset_integrity(config=config, mode=mode, logger=logger, stage="retrieval_resource_precheck")

    paths = config["paths"]
    llm_cfg = config["training_llm"]
    graph_cfg = config["graph_pipeline"]

    taxonomy = load_taxonomy_file(config["taxonomy"]["taxonomy_file"])
    all_labeled = sorted(_load_models(Path(paths["labeled_complaints"]), NormalizedComplaint), key=lambda item: item.complaint_id)
    train_records = sorted(_load_models(Path(paths["train_split"]), NormalizedComplaint), key=lambda item: item.complaint_id)
    val_records = sorted(_load_models(Path(paths["val_split"]), NormalizedComplaint), key=lambda item: item.complaint_id)
    test_records = sorted(_load_models(Path(paths["test_split"]), NormalizedComplaint), key=lambda item: item.complaint_id)
    hard_test_records = sorted(_load_models(Path(paths["hard_test_split"]), NormalizedComplaint), key=lambda item: item.complaint_id)
    steps = sorted(_load_models(Path(paths["solution_steps_jsonl"]), SolutionStep), key=lambda item: item.step_id)
    kb_items = sorted(_load_models(Path(paths["kb_jsonl"]), KBParagraph), key=lambda item: item.paragraph_id)
    links = sorted(_load_models(Path(paths["step_kb_links_jsonl"]), StepKBLink), key=lambda item: item.step_id)

    if mode == "SMOKE":
        limit = int(llm_cfg["smoke"]["retrieval_fit_limit"])
        fit_records = all_labeled[: max(1, min(limit, len(all_labeled)))]
    else:
        fit_records = all_labeled

    embedder = HashingTextEmbedder(
        dimension=int(graph_cfg["embeddings"]["dimension"]),
        ngram_min=int(graph_cfg["embeddings"]["ngram_min"]),
        ngram_max=int(graph_cfg["embeddings"]["ngram_max"]),
    )
    cache = EmbeddingCache(paths["embeddings_dir"])
    text_payloads = _build_text_payloads(fit_records, steps, kb_items, taxonomy)

    embeddings: dict[str, tuple[list[str], np.ndarray]] = {}
    for key, (ids, texts) in text_payloads.items():
        vectors = cache.get_or_compute(
            key=f"train_llm_{key}",
            ids=ids,
            texts=texts,
            embedder=embedder,
            force_recompute=bool(llm_cfg["dataset"].get("force_recompute_retrieval_embeddings", False)),
        )
        embeddings[key] = (ids, vectors)

    step_prior_weight = {step.step_id: 0.7 for step in steps}
    retrieval_resources = build_retrieval_resources(
        taxonomy=taxonomy,
        complaints=fit_records,
        steps=steps,
        kb_items=kb_items,
        links=links,
        embeddings=embeddings,
        embedder_callable=embedder.embed,
        config={
            "taxonomy_assignment": config["taxonomy"]["assignment"],
            "seed": int(config["reproducibility"]["seed"]),
            "alpha": float(graph_cfg["retrieval"]["alpha"]),
            "beta": float(graph_cfg["retrieval"]["beta"]),
            "gamma": float(graph_cfg["retrieval"]["gamma"]),
            "delta": float(graph_cfg["retrieval"]["delta"]),
            "lambda_gnn": 0.0,
            "top_steps": int(llm_cfg["dataset"]["task2_top_steps"]),
            "min_steps": int(llm_cfg["dataset"]["task2_min_steps"]),
            "max_evidence": int(llm_cfg["dataset"]["task2_max_evidence"]),
            "escalation_threshold": float(graph_cfg["retrieval"]["escalation_threshold"]),
        },
        step_prior_weight=step_prior_weight,
        step_gnn_embeddings=None,
        category_gnn_embeddings=None,
    )

    log_event(
        logger,
        "INFO",
        "train_llm_retrieval_resources_ready",
        {
            "fit_records": len(fit_records),
            "train": len(train_records),
            "val": len(val_records),
            "test": len(test_records),
            "hard_test": len(hard_test_records),
            "steps": len(steps),
            "kb": len(kb_items),
        },
    )
    return retrieval_resources, train_records, val_records, test_records, hard_test_records


def _example_id(prefix: str, complaint_id: str, dataset_version: str) -> str:
    digest = hashlib.sha256(f"{prefix}|{complaint_id}|{dataset_version}".encode("utf-8")).hexdigest()[:16]
    return f"{prefix}.{digest}"


def _default_retrieval_pack_provider(
    record: NormalizedComplaint,
    resources: RetrievalResources,
    include_debug: bool,
) -> tuple[Any, dict[str, Any]]:
    return retrieve_evidence_pack(
        complaint_text=record.complaint_text_clean,
        resources=resources,
        request_id=f"TASK2:{record.complaint_id}",
        brand=record.brand_slug,
        include_debug=include_debug,
    )


def build_task2_examples(
    *,
    records: list[NormalizedComplaint],
    split: str,
    resources: RetrievalResources,
    dataset_version: str,
    min_steps: int,
    max_steps: int,
    include_debug: bool,
    retrieval_pack_provider: Callable[[NormalizedComplaint, RetrievalResources, bool], tuple[Any, dict[str, Any]]] | None = None,
) -> list[Task2SFTExample]:
    provider = retrieval_pack_provider or _default_retrieval_pack_provider
    system_prompt = build_strict_system_prompt_tr()
    examples: list[Task2SFTExample] = []

    for record in sorted(records, key=lambda item: item.complaint_id):
        pack, _ = provider(record, resources, include_debug)
        target = render_deterministic_response(pack, min_steps=min_steps, max_steps=max_steps)
        allowed_steps = [item.step_id for item in pack.top_steps[:max_steps]]
        allowed_evidence = sorted(
            {
                evidence_id
                for step in pack.top_steps[:max_steps]
                for evidence_id in step.evidence_ids
            }
        )
        example = Task2SFTExample(
            example_id=_example_id("TASK2", record.complaint_id, dataset_version),
            complaint_id=record.complaint_id,
            split=split,
            system_prompt=system_prompt,
            user_message=record.complaint_text_clean,
            assistant_message=target,
            normalized_category=pack.normalized_category,
            category_confidence=pack.category_confidence,
            allowed_step_ids=allowed_steps,
            allowed_evidence_ids=allowed_evidence,
            messages=[
                ChatMessage(role="system", content=system_prompt),
                ChatMessage(role="user", content=record.complaint_text_clean),
                ChatMessage(role="assistant", content=target),
            ],
            source_hash_sha256=record.source_hash_sha256,
            dataset_version=dataset_version,
        )
        examples.append(example)

    return examples


def build_task1_examples(
    *,
    records: list[NormalizedComplaint],
    split: str,
    dataset_version: str,
) -> list[Task1IntentExample]:
    instruction = "Aşağıdaki şikayeti verilen telekom taksonomisine göre tek bir category_id ile etiketle."
    examples: list[Task1IntentExample] = []
    for record in sorted(records, key=lambda item: item.complaint_id):
        examples.append(
            Task1IntentExample(
                example_id=_example_id("TASK1", record.complaint_id, dataset_version),
                complaint_id=record.complaint_id,
                split=split,
                instruction=instruction,
                user_message=record.complaint_text_clean,
                assistant_message=record.normalized_category,
                label_category_id=record.normalized_category,
                source_hash_sha256=record.source_hash_sha256,
                dataset_version=dataset_version,
            )
        )
    return examples


def build_and_write_training_datasets(
    *,
    config: dict[str, Any],
    mode: str,
    logger: logging.Logger,
) -> dict[str, Any]:
    llm_cfg = config["training_llm"]
    paths = config["paths"]
    dataset_version = str(llm_cfg["dataset"]["version"])

    resources, train_records, val_records, _, _ = build_retrieval_resources_from_artifacts(
        config=config,
        mode=mode,
        logger=logger,
    )

    if mode == "SMOKE":
        train_records = train_records[: int(llm_cfg["smoke"]["task2_train_limit"])]
        val_records = val_records[: int(llm_cfg["smoke"]["task2_val_limit"])]

    task2_train = build_task2_examples(
        records=train_records,
        split="train",
        resources=resources,
        dataset_version=dataset_version,
        min_steps=int(llm_cfg["dataset"]["task2_min_steps"]),
        max_steps=int(llm_cfg["dataset"]["task2_top_steps"]),
        include_debug=False,
    )
    task2_val = build_task2_examples(
        records=val_records,
        split="val",
        resources=resources,
        dataset_version=dataset_version,
        min_steps=int(llm_cfg["dataset"]["task2_min_steps"]),
        max_steps=int(llm_cfg["dataset"]["task2_top_steps"]),
        include_debug=False,
    )

    _write_jsonl(Path(paths["task2_sft_train"]), task2_train)
    _write_jsonl(Path(paths["task2_sft_val"]), task2_val)

    task1_train_count = 0
    task1_val_count = 0
    if bool(config["training"].get("use_llm_for_intent", False)):
        task1_train = build_task1_examples(records=train_records, split="train", dataset_version=dataset_version)
        task1_val = build_task1_examples(records=val_records, split="val", dataset_version=dataset_version)
        _write_jsonl(Path(paths["task1_intent_train"]), task1_train)
        _write_jsonl(Path(paths["task1_intent_val"]), task1_val)
        task1_train_count = len(task1_train)
        task1_val_count = len(task1_val)

    stats = {
        "task2_train": len(task2_train),
        "task2_val": len(task2_val),
        "task1_train": task1_train_count,
        "task1_val": task1_val_count,
        "dataset_version": dataset_version,
    }
    log_event(logger, "INFO", "train_llm_dataset_built", stats)
    return stats
