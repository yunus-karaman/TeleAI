from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass


MINHASH_PRIME = (1 << 61) - 1


@dataclass(frozen=True)
class DuplicateCluster:
    cluster_id: str
    member_ids: list[str]
    canonical_id: str


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, index: int) -> int:
        while self.parent[index] != index:
            self.parent[index] = self.parent[self.parent[index]]
            index = self.parent[index]
        return index

    def union(self, left: int, right: int) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left == root_right:
            return
        if self.rank[root_left] < self.rank[root_right]:
            self.parent[root_left] = root_right
        elif self.rank[root_left] > self.rank[root_right]:
            self.parent[root_right] = root_left
        else:
            self.parent[root_right] = root_left
            self.rank[root_left] += 1


def _stable_hash_64(text: str) -> int:
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=False)


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9çğıöşü]+", text.lower())


def _build_shingles(text: str, shingle_size: int) -> set[str]:
    tokens = _tokenize(text)
    if not tokens:
        return set()
    if len(tokens) < shingle_size:
        return {" ".join(tokens)}
    return {" ".join(tokens[index : index + shingle_size]) for index in range(0, len(tokens) - shingle_size + 1)}


def _build_minhash_signature(shingles: set[str], salts: list[int]) -> tuple[int, ...]:
    if not shingles:
        return tuple(0 for _ in salts)
    base_hashes = [_stable_hash_64(item) for item in shingles]
    signature: list[int] = []
    for salt in salts:
        min_hash = min((value ^ salt) % MINHASH_PRIME for value in base_hashes)
        signature.append(min_hash)
    return tuple(signature)


def _jaccard_similarity(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    intersection = len(left & right)
    union = len(left | right)
    return intersection / union if union else 0.0


def cluster_near_duplicates(
    complaint_ids: list[str],
    texts: list[str],
    shingle_size: int,
    num_perm: int,
    bands: int,
    similarity_threshold: float,
    random_seed: int = 42,
) -> list[DuplicateCluster]:
    if len(complaint_ids) != len(texts):
        raise ValueError("complaint_ids and texts must have the same length")
    if bands <= 0 or num_perm <= 0:
        return []
    if num_perm % bands != 0:
        raise ValueError("num_perm must be divisible by bands")

    rows_per_band = num_perm // bands
    randomizer = random.Random(random_seed)
    salts = [randomizer.getrandbits(61) for _ in range(num_perm)]

    shingles_per_record = [_build_shingles(text, shingle_size=shingle_size) for text in texts]
    signatures = [_build_minhash_signature(shingles, salts=salts) for shingles in shingles_per_record]

    buckets: dict[tuple[int, tuple[int, ...]], list[int]] = {}
    for index, signature in enumerate(signatures):
        for band in range(bands):
            start = band * rows_per_band
            end = start + rows_per_band
            key = (band, signature[start:end])
            buckets.setdefault(key, []).append(index)

    candidate_pairs: set[tuple[int, int]] = set()
    for bucket_members in buckets.values():
        if len(bucket_members) < 2:
            continue
        sorted_members = sorted(set(bucket_members))
        for left_index, left_member in enumerate(sorted_members):
            for right_member in sorted_members[left_index + 1 :]:
                candidate_pairs.add((left_member, right_member))

    union_find = _UnionFind(size=len(complaint_ids))
    for left, right in sorted(candidate_pairs):
        similarity = _jaccard_similarity(shingles_per_record[left], shingles_per_record[right])
        if similarity >= similarity_threshold:
            union_find.union(left, right)

    groups: dict[int, list[int]] = {}
    for index in range(len(complaint_ids)):
        root = union_find.find(index)
        groups.setdefault(root, []).append(index)

    clusters: list[DuplicateCluster] = []
    cluster_counter = 1
    for members in sorted(groups.values(), key=lambda idxs: min(complaint_ids[i] for i in idxs)):
        if len(members) < 2:
            continue
        member_ids = sorted(complaint_ids[index] for index in members)
        canonical_id = member_ids[0]
        cluster_id = f"dup_cluster_{cluster_counter:05d}"
        clusters.append(
            DuplicateCluster(
                cluster_id=cluster_id,
                member_ids=member_ids,
                canonical_id=canonical_id,
            )
        )
        cluster_counter += 1

    return clusters

