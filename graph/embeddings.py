from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer


class TextEmbedder(Protocol):
    name: str
    dimension: int

    def embed(self, texts: list[str]) -> np.ndarray:
        ...


@dataclass(frozen=True)
class HashingTextEmbedder:
    dimension: int = 768
    ngram_min: int = 1
    ngram_max: int = 2

    @property
    def name(self) -> str:
        return "hashing_tfidf"

    def embed(self, texts: list[str]) -> np.ndarray:
        vectorizer = HashingVectorizer(
            n_features=self.dimension,
            alternate_sign=False,
            norm="l2",
            lowercase=True,
            ngram_range=(self.ngram_min, self.ngram_max),
        )
        matrix = vectorizer.transform([text or "" for text in texts])
        return matrix.astype(np.float32).toarray()


def _ids_hash(ids: list[str]) -> str:
    payload = "\n".join(ids)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class EmbeddingCache:
    def __init__(self, base_dir: str | Path) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def get_or_compute(
        self,
        *,
        key: str,
        ids: list[str],
        texts: list[str],
        embedder: TextEmbedder,
        force_recompute: bool = False,
    ) -> np.ndarray:
        if len(ids) != len(texts):
            raise ValueError("ids and texts length must match.")
        target = self.base_dir / f"{key}.npz"
        metadata = {
            "key": key,
            "embedder": embedder.name,
            "dimension": embedder.dimension,
            "ids_hash": _ids_hash(ids),
            "count": len(ids),
        }
        cache_recompute_reason: str | None = None

        if target.exists() and not force_recompute:
            try:
                loaded = np.load(target, allow_pickle=False)
                loaded_vectors = loaded["vectors"]
                loaded_ids = loaded["ids"].astype(str).tolist()
                loaded_meta = json.loads(str(loaded["meta"][0]))
                if loaded_meta == metadata and loaded_ids == ids:
                    return loaded_vectors.astype(np.float32)
                cache_recompute_reason = "cache_metadata_or_ids_mismatch"
            except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
                cache_recompute_reason = f"cache_load_error:{error}"

        if cache_recompute_reason:
            metadata["cache_recompute_reason"] = cache_recompute_reason

        vectors = embedder.embed(texts=texts)
        ids_arr = np.array(ids, dtype=np.str_)
        meta_arr = np.array([json.dumps(metadata, ensure_ascii=False)], dtype=np.str_)
        np.savez_compressed(
            target,
            ids=ids_arr,
            vectors=vectors.astype(np.float32),
            meta=meta_arr,
        )
        return vectors.astype(np.float32)
