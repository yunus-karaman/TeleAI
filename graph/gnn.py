from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from data.schemas import GraphEdge


@dataclass
class GNNResult:
    node_ids: list[str]
    embeddings: np.ndarray
    metadata: dict[str, Any]


def run_lightweight_gnn(
    *,
    node_ids: list[str],
    base_embeddings: np.ndarray,
    edges: list[GraphEdge],
    epochs: int,
    self_weight: float,
    neighbor_weight: float,
    convergence_tol: float,
) -> GNNResult:
    if len(node_ids) != base_embeddings.shape[0]:
        raise ValueError("node_ids length and base_embeddings rows must match.")
    id_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}
    neighbors: list[list[tuple[int, float]]] = [[] for _ in node_ids]

    for edge in edges:
        src_idx = id_to_idx.get(edge.source_node_id)
        dst_idx = id_to_idx.get(edge.target_node_id)
        if src_idx is None or dst_idx is None:
            continue
        weight = float(edge.weight)
        neighbors[src_idx].append((dst_idx, weight))
        neighbors[dst_idx].append((src_idx, weight))

    embeddings = base_embeddings.astype(np.float32).copy()
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.maximum(norms, 1e-8)

    converged_epoch = epochs
    for epoch in range(1, epochs + 1):
        updated = embeddings.copy()
        for node_idx, node_neighbors in enumerate(neighbors):
            if not node_neighbors:
                continue
            neighbor_vectors = []
            for nbr_idx, weight in node_neighbors:
                neighbor_vectors.append(weight * embeddings[nbr_idx])
            mean_neighbor = np.mean(np.stack(neighbor_vectors, axis=0), axis=0)
            combined = (self_weight * embeddings[node_idx]) + (neighbor_weight * mean_neighbor)
            norm = np.linalg.norm(combined)
            updated[node_idx] = combined / max(norm, 1e-8)
        delta = float(np.mean(np.linalg.norm(updated - embeddings, axis=1)))
        embeddings = updated
        if delta <= convergence_tol:
            converged_epoch = epoch
            break

    metadata = {
        "epochs_requested": epochs,
        "epochs_run": converged_epoch,
        "self_weight": self_weight,
        "neighbor_weight": neighbor_weight,
        "convergence_tol": convergence_tol,
        "node_count": len(node_ids),
        "dim": int(embeddings.shape[1]),
    }
    return GNNResult(node_ids=node_ids, embeddings=embeddings, metadata=metadata)


def save_gnn_result(result: GNNResult, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target,
        node_ids=np.array(result.node_ids, dtype=object),
        embeddings=result.embeddings.astype(np.float32),
        metadata=np.array(json.dumps(result.metadata, ensure_ascii=False), dtype=object),
    )

