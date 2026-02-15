from __future__ import annotations

import time
import uuid
import re
from dataclasses import dataclass
from typing import Any

import numpy as np

from data.schemas import EvidencePack, KBParagraph, NormalizedComplaint, SolutionStep
from solution_steps.generator import StepKBLink
from taxonomy.assignment import HybridTaxonomyAssigner
from taxonomy.schema import TaxonomyFile


TOKEN_PATTERN = re.compile(r"[a-z0-9çğıöşü]{2,}", flags=re.IGNORECASE)


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom == 0.0:
        return 0.0
    return float(np.dot(left, right) / denom)


def _tokenize(text: str) -> set[str]:
    return set(TOKEN_PATTERN.findall((text or "").lower()))


@dataclass
class RetrievalResources:
    taxonomy: TaxonomyFile
    taxonomy_assigner: HybridTaxonomyAssigner
    steps: list[SolutionStep]
    kb_items: list[KBParagraph]
    links: list[StepKBLink]
    step_embeddings: np.ndarray
    kb_embeddings: np.ndarray
    category_embeddings: np.ndarray
    complaint_embeddings: np.ndarray
    complaint_records: list[NormalizedComplaint]
    id_to_step_index: dict[str, int]
    id_to_kb_index: dict[str, int]
    category_to_step_ids: dict[str, list[str]]
    step_to_evidence_ids: dict[str, list[str]]
    step_prior_weight: dict[str, float]
    category_centroids: dict[str, np.ndarray]
    step_gnn_embeddings: dict[str, np.ndarray] | None
    category_gnn_embeddings: dict[str, np.ndarray] | None
    embed_text: Any
    config: dict[str, Any]


def build_retrieval_resources(
    *,
    taxonomy: TaxonomyFile,
    complaints: list[NormalizedComplaint],
    steps: list[SolutionStep],
    kb_items: list[KBParagraph],
    links: list[StepKBLink],
    embeddings: dict[str, tuple[list[str], np.ndarray]],
    embedder_callable: Any,
    config: dict[str, Any],
    step_prior_weight: dict[str, float],
    step_gnn_embeddings: dict[str, np.ndarray] | None = None,
    category_gnn_embeddings: dict[str, np.ndarray] | None = None,
) -> RetrievalResources:
    taxonomy_assigner = HybridTaxonomyAssigner(
        taxonomy=taxonomy,
        config=config["taxonomy_assignment"],
        seed=int(config["seed"]),
    )
    taxonomy_assigner.fit([record.complaint_text_clean for record in complaints])

    step_ids, step_embeddings = embeddings["steps"]
    kb_ids, kb_embeddings = embeddings["kb"]
    category_ids, category_embeddings = embeddings["categories"]
    complaint_ids, complaint_embeddings = embeddings["complaints"]

    id_to_step_index = {step_id: index for index, step_id in enumerate(step_ids)}
    id_to_kb_index = {paragraph_id: index for index, paragraph_id in enumerate(kb_ids)}
    category_index = {category_id: idx for idx, category_id in enumerate(category_ids)}

    category_to_step_ids: dict[str, list[str]] = {}
    for step in sorted(steps, key=lambda item: item.step_id):
        category_to_step_ids.setdefault(step.category_id, []).append(step.step_id)

    step_to_evidence_ids: dict[str, list[str]] = {}
    for link in sorted(links, key=lambda item: item.step_id):
        ordered = sorted({evidence_id for evidence_id in link.evidence_ids})
        step_to_evidence_ids[link.step_id] = ordered

    complaint_by_category: dict[str, list[np.ndarray]] = {}
    complaint_id_to_index = {complaint_id: index for index, complaint_id in enumerate(complaint_ids)}
    for record in complaints:
        index = complaint_id_to_index.get(record.complaint_id)
        if index is None:
            continue
        complaint_by_category.setdefault(record.normalized_category, []).append(complaint_embeddings[index])

    category_centroids: dict[str, np.ndarray] = {}
    for category in taxonomy.categories:
        cid = category.category_id
        vectors = complaint_by_category.get(cid, [])
        if vectors:
            centroid = np.mean(np.stack(vectors, axis=0), axis=0)
        else:
            centroid = category_embeddings[category_index[cid]]
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm
        category_centroids[cid] = centroid.astype(np.float32)

    return RetrievalResources(
        taxonomy=taxonomy,
        taxonomy_assigner=taxonomy_assigner,
        steps=steps,
        kb_items=kb_items,
        links=links,
        step_embeddings=step_embeddings,
        kb_embeddings=kb_embeddings,
        category_embeddings=category_embeddings,
        complaint_embeddings=complaint_embeddings,
        complaint_records=complaints,
        id_to_step_index=id_to_step_index,
        id_to_kb_index=id_to_kb_index,
        category_to_step_ids=category_to_step_ids,
        step_to_evidence_ids=step_to_evidence_ids,
        step_prior_weight=step_prior_weight,
        category_centroids=category_centroids,
        step_gnn_embeddings=step_gnn_embeddings,
        category_gnn_embeddings=category_gnn_embeddings,
        embed_text=embedder_callable,
        config=config,
    )


def retrieve_evidence_pack(
    *,
    complaint_text: str,
    resources: RetrievalResources,
    request_id: str | None = None,
    brand: str | None = None,
    time_bucket: str | None = None,
    include_debug: bool = False,
) -> tuple[EvidencePack, dict[str, Any]]:
    start = time.perf_counter()
    request_id = request_id or f"REQ-{uuid.uuid4().hex[:12]}"
    config = resources.config
    query_embedding = resources.embed_text([complaint_text])[0]
    query_tokens = _tokenize(complaint_text)

    category_assignment = resources.taxonomy_assigner.assign(complaint_text)
    requested_category_id = category_assignment.normalized_category
    category_id = requested_category_id
    category_conf = category_assignment.confidence_score
    category_steps = resources.category_to_step_ids.get(category_id, [])
    fallback_category_used = False
    if not category_steps:
        ranked_categories = sorted(
            category_assignment.combined_scores.items(),
            key=lambda item: (-float(item[1]), item[0]),
        )
        for candidate_category, _ in ranked_categories:
            candidate_steps = resources.category_to_step_ids.get(candidate_category, [])
            if candidate_steps:
                category_id = candidate_category
                category_steps = candidate_steps
                fallback_category_used = candidate_category != requested_category_id
                break

    alpha = float(config["alpha"])
    beta = float(config["beta"])
    gamma = float(config["gamma"])
    delta = float(config["delta"])
    lambda_gnn = float(config["lambda_gnn"])
    top_n = int(config["top_steps"])
    min_n = int(config["min_steps"])
    max_evidence = int(config["max_evidence"])

    centroid = resources.category_centroids.get(category_id)
    cluster_signal = _cosine(query_embedding, centroid) if centroid is not None else 0.0

    scored_steps: list[tuple[str, float, dict[str, float]]] = []
    for step_id in category_steps:
        evidence_ids = resources.step_to_evidence_ids.get(step_id, [])
        if not evidence_ids:
            continue
        step_idx = resources.id_to_step_index[step_id]
        step = resources.steps[step_idx]
        step_vec = resources.step_embeddings[step_idx]
        sim_score = _cosine(query_embedding, step_vec)

        step_tag_tokens = set()
        for tag in step.tags:
            step_tag_tokens.update(_tokenize(tag.replace("_", " ")))
        tag_overlap = (len(query_tokens & step_tag_tokens) / len(step_tag_tokens)) if step_tag_tokens else 0.0
        prior = resources.step_prior_weight.get(step_id, 0.5)

        gnn_score = 0.0
        if resources.step_gnn_embeddings is not None and resources.category_gnn_embeddings is not None:
            step_gnn = resources.step_gnn_embeddings.get(step_id)
            category_gnn = resources.category_gnn_embeddings.get(category_id)
            if step_gnn is not None and category_gnn is not None:
                query_gnn = (0.7 * query_embedding) + (0.3 * category_gnn)
                gnn_score = _cosine(query_gnn, step_gnn)

        total = (alpha * sim_score) + (beta * tag_overlap) + (gamma * prior) + (delta * cluster_signal) + (
            lambda_gnn * gnn_score
        )
        scored_steps.append(
            (
                step_id,
                float(total),
                {
                    "sim": float(sim_score),
                    "tag_overlap": float(tag_overlap),
                    "prior": float(prior),
                    "cluster_signal": float(cluster_signal),
                    "gnn": float(gnn_score),
                },
            )
        )

    if not scored_steps:
        # Deterministic hard fallback: pick globally valid steps with evidence.
        for step in sorted(resources.steps, key=lambda item: item.step_id):
            evidence_ids = resources.step_to_evidence_ids.get(step.step_id, [])
            if not evidence_ids:
                continue
            scored_steps.append(
                (
                    step.step_id,
                    0.0,
                    {
                        "sim": 0.0,
                        "tag_overlap": 0.0,
                        "prior": float(resources.step_prior_weight.get(step.step_id, 0.0)),
                        "cluster_signal": float(cluster_signal),
                        "gnn": 0.0,
                    },
                )
            )
        if scored_steps:
            category_id = resources.steps[resources.id_to_step_index[scored_steps[0][0]]].category_id
            fallback_category_used = True

    scored_steps.sort(key=lambda item: (-item[1], item[0]))
    selected = scored_steps[: max(min_n, top_n)]
    if len(selected) < min_n:
        selected = scored_steps[:min_n]

    kb_index_by_id = resources.id_to_kb_index
    kb_by_id = {item.paragraph_id: item for item in resources.kb_items}
    top_steps_payload: list[EvidencePack.TopStepItem] = []
    evidence_scores: dict[str, float] = {}
    step_debug: dict[str, Any] = {}

    for step_id, score, components in selected:
        step = resources.steps[resources.id_to_step_index[step_id]]
        candidate_evidence = resources.step_to_evidence_ids.get(step_id, [])
        ranked_local: list[tuple[str, float]] = []
        for evidence_id in candidate_evidence:
            kb_idx = kb_index_by_id[evidence_id]
            kb_vec = resources.kb_embeddings[kb_idx]
            kb_item = kb_by_id[evidence_id]
            sim_kb = _cosine(query_embedding, kb_vec)
            evidence_score = (0.6 * float(kb_item.confidence)) + (0.4 * sim_kb)
            ranked_local.append((evidence_id, float(evidence_score)))
            evidence_scores[evidence_id] = max(evidence_scores.get(evidence_id, 0.0), float(evidence_score))
        ranked_local.sort(key=lambda item: (-item[1], item[0]))
        chosen_evidence = [evidence_id for evidence_id, _ in ranked_local[:2]]
        if not chosen_evidence and candidate_evidence:
            chosen_evidence = [candidate_evidence[0]]

        top_steps_payload.append(
            EvidencePack.TopStepItem(
                step_id=step.step_id,
                title_tr=step.title_tr,
                level=step.level,
                instructions_tr=step.instructions_tr,
                evidence_ids=chosen_evidence,
                step_score=round(max(0.0, min(1.0, score)), 6),
            )
        )
        step_debug[step.step_id] = {
            "score_components": components,
            "raw_score": score,
            "chosen_evidence": chosen_evidence,
        }

    ranked_evidence = sorted(evidence_scores.items(), key=lambda item: (-item[1], item[0]))[:max_evidence]
    evidence_payload: list[EvidencePack.EvidenceItem] = []
    for evidence_id, score in ranked_evidence:
        kb_item = kb_by_id[evidence_id]
        evidence_payload.append(
            EvidencePack.EvidenceItem(
                paragraph_id=evidence_id,
                text_tr=kb_item.text_tr,
                confidence=round(max(0.0, min(1.0, score)), 6),
            )
        )

    if not top_steps_payload:
        raise RuntimeError("Retrieval produced zero top steps; solution dataset integrity is inconsistent.")
    if not evidence_payload:
        raise RuntimeError("Retrieval produced zero evidence rows for selected steps.")

    escalation_threshold = float(config["escalation_threshold"])
    top_level = top_steps_payload[0].level if top_steps_payload else "L1"
    top_step = resources.steps[resources.id_to_step_index[top_steps_payload[0].step_id]] if top_steps_payload else None
    should_escalate = category_conf < escalation_threshold or top_level == "L3"
    escalation_unit = top_step.escalation_unit if top_step is not None else "GENERAL_SUPPORT"
    reason = (
        "Kategori guveni dusuk oldugu icin resmi destek takibi onerilir."
        if category_conf < escalation_threshold
        else "Yuksek seviye adim gerektigi icin resmi destek birimine aktarim onerilir."
        if top_level == "L3"
        else "Self-servis adimlar ile devam edilebilir; sorun surerse destek birimine aktarilir."
    )
    threshold_signals = []
    if category_conf < escalation_threshold:
        threshold_signals.append("LOW_CATEGORY_CONFIDENCE")
    if top_level == "L3":
        threshold_signals.append("L3_STEP_SELECTED")
    if brand:
        threshold_signals.append(f"BRAND_CONTEXT:{brand}")
    if time_bucket:
        threshold_signals.append(f"TIME_CONTEXT:{time_bucket}")

    debug_payload = None
    if include_debug:
        debug_payload = {
            "category_assignment": {
                "category": category_id,
                "confidence": category_conf,
                "reason": category_assignment.assignment_reason,
                "requested_category": requested_category_id,
                "fallback_category_used": fallback_category_used,
            },
            "step_debug": step_debug,
            "timing_ms": None,
        }

    pack = EvidencePack(
        request_id=request_id,
        normalized_category=category_id,
        category_confidence=round(max(0.0, min(1.0, category_conf)), 6),
        top_steps=top_steps_payload,
        evidence=evidence_payload,
        escalation_suggestion=EvidencePack.EscalationSuggestion(
            unit=escalation_unit if should_escalate else "GENERAL_SUPPORT",
            reason=reason,
            threshold_signals=threshold_signals,
        ),
        retrieval_debug=debug_payload,
    )

    elapsed_ms = (time.perf_counter() - start) * 1000.0
    if include_debug and pack.retrieval_debug is not None:
        pack.retrieval_debug["timing_ms"] = round(elapsed_ms, 3)

    telemetry = {
        "latency_ms": elapsed_ms,
        "category_confidence": pack.category_confidence,
        "step_count": len(pack.top_steps),
        "evidence_count": len(pack.evidence),
        "top_level": top_level,
        "escalation": should_escalate,
    }
    return pack, telemetry
