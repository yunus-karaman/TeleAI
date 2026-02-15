from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from taxonomy.schema import TaxonomyCategory, TaxonomyFile


TOKEN_PATTERN = re.compile(r"[a-z0-9çğıöşü]{2,}", flags=re.IGNORECASE)


@dataclass(frozen=True)
class AssignmentResult:
    normalized_category: str
    confidence_score: float
    assignment_reason: str
    needs_review: bool
    rule_top_category: str
    embedding_top_category: str
    combined_scores: dict[str, float]
    rule_scores: dict[str, float]
    embedding_scores: dict[str, float]


def _tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall((text or "").lower())


def _normalize_non_negative_scores(scores: dict[str, float]) -> dict[str, float]:
    clipped = {category: max(0.0, value) for category, value in scores.items()}
    max_score = max(clipped.values()) if clipped else 0.0
    if max_score <= 0.0:
        return {category: 0.0 for category in clipped}
    return {category: round(value / max_score, 8) for category, value in clipped.items()}


def _compute_rule_scores(
    text: str,
    categories: list[TaxonomyCategory],
    keyword_weight: float,
    negative_weight: float,
    example_weight: float,
) -> tuple[dict[str, float], dict[str, list[str]]]:
    lowered = (text or "").lower()
    tokens = _tokenize(lowered)
    token_counts = Counter(tokens)
    token_norm = math.sqrt(len(tokens) + 1)

    raw_scores: dict[str, float] = {}
    matched_terms: dict[str, list[str]] = {}
    for category in categories:
        score = 0.0
        hits: list[str] = []
        for keyword in category.keywords_tr:
            keyword_lower = keyword.lower()
            if " " in keyword_lower:
                count = lowered.count(keyword_lower)
            else:
                count = token_counts.get(keyword_lower, 0)
            if count > 0:
                score += keyword_weight * count
                hits.append(f"+{keyword_lower}:{count}")

        for phrase in category.example_phrases_tr:
            phrase_lower = phrase.lower()
            if phrase_lower in lowered:
                score += example_weight
                hits.append(f"+example:{phrase_lower[:28]}")

        for neg_keyword in category.negative_keywords_tr:
            neg_lower = neg_keyword.lower()
            if " " in neg_lower:
                neg_count = lowered.count(neg_lower)
            else:
                neg_count = token_counts.get(neg_lower, 0)
            if neg_count > 0:
                score -= negative_weight * neg_count
                hits.append(f"-{neg_lower}:{neg_count}")

        score = score / token_norm if token_norm > 0 else 0.0
        raw_scores[category.category_id] = round(score, 8)
        matched_terms[category.category_id] = hits

    return raw_scores, matched_terms


class HybridTaxonomyAssigner:
    def __init__(self, taxonomy: TaxonomyFile, config: dict[str, Any], seed: int) -> None:
        self.taxonomy = taxonomy
        self.categories = taxonomy.categories
        self.category_ids = [category.category_id for category in self.categories]
        self.seed = seed

        self.rule_weight = float(config["rule_weight"])
        self.embedding_weight = float(config["embedding_weight"])
        self.keyword_weight = float(config["keyword_weight"])
        self.negative_weight = float(config["negative_weight"])
        self.example_weight = float(config["example_weight"])
        self.min_confidence = float(config["min_confidence"])
        self.low_confidence_policy = str(config["low_confidence_policy"]).lower()
        self.review_margin_threshold = float(config["review_margin_threshold"])
        self.max_features = int(config["embedding"]["max_features"])
        self.ngram_min = int(config["embedding"]["ngram_min"])
        self.ngram_max = int(config["embedding"]["ngram_max"])
        self.min_df = int(config["embedding"]["min_df"])

        self.vectorizer: TfidfVectorizer | None = None
        self.category_matrix = None

    def fit(self, complaint_texts: list[str]) -> None:
        category_documents = [self._build_category_document(category) for category in self.categories]
        fit_corpus = [text or "" for text in complaint_texts] + category_documents
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            sublinear_tf=True,
            min_df=self.min_df,
            max_features=self.max_features,
            ngram_range=(self.ngram_min, self.ngram_max),
        )
        self.vectorizer.fit(fit_corpus)
        self.category_matrix = self.vectorizer.transform(category_documents)

    def assign(self, text: str) -> AssignmentResult:
        if self.vectorizer is None or self.category_matrix is None:
            raise RuntimeError("HybridTaxonomyAssigner.fit must be called before assign.")

        rule_raw, matched_terms = _compute_rule_scores(
            text=text,
            categories=self.categories,
            keyword_weight=self.keyword_weight,
            negative_weight=self.negative_weight,
            example_weight=self.example_weight,
        )
        rule_scores = _normalize_non_negative_scores(rule_raw)

        vector = self.vectorizer.transform([text or ""])
        embedding_values = cosine_similarity(vector, self.category_matrix)[0]
        embedding_raw = {category_id: float(score) for category_id, score in zip(self.category_ids, embedding_values)}
        embedding_scores = _normalize_non_negative_scores(embedding_raw)

        combined_scores = {
            category_id: round(
                (self.rule_weight * rule_scores.get(category_id, 0.0))
                + (self.embedding_weight * embedding_scores.get(category_id, 0.0)),
                8,
            )
            for category_id in self.category_ids
        }

        ranked = sorted(
            self.category_ids,
            key=lambda category_id: (
                -combined_scores[category_id],
                -rule_scores.get(category_id, 0.0),
                category_id,
            ),
        )
        top_category = ranked[0]
        second_score = combined_scores[ranked[1]] if len(ranked) > 1 else 0.0
        top_score = combined_scores[top_category]
        margin = max(0.0, top_score - second_score)
        confidence_score = max(0.0, min(1.0, (0.75 * top_score) + (0.25 * margin)))

        needs_review = False
        final_category = top_category
        if confidence_score < self.min_confidence:
            needs_review = True
            if self.low_confidence_policy == "other":
                final_category = "OTHER"
            elif self.low_confidence_policy == "needs_review":
                final_category = top_category
            else:
                final_category = "OTHER"
        elif margin < self.review_margin_threshold:
            needs_review = True

        top_rule_category = max(self.category_ids, key=lambda cid: (rule_scores.get(cid, 0.0), cid))
        top_embedding_category = max(self.category_ids, key=lambda cid: (embedding_scores.get(cid, 0.0), cid))
        top_terms = matched_terms.get(top_category, [])[:4]
        reason = (
            f"top={top_category}; score={top_score:.4f}; rule={rule_scores.get(top_category,0.0):.4f}; "
            f"embed={embedding_scores.get(top_category,0.0):.4f}; margin={margin:.4f}; terms={','.join(top_terms)}"
        )

        return AssignmentResult(
            normalized_category=final_category,
            confidence_score=round(confidence_score, 6),
            assignment_reason=reason,
            needs_review=needs_review,
            rule_top_category=top_rule_category,
            embedding_top_category=top_embedding_category,
            combined_scores=combined_scores,
            rule_scores=rule_scores,
            embedding_scores=embedding_scores,
        )

    @staticmethod
    def _build_category_document(category: TaxonomyCategory) -> str:
        chunks = [
            category.title_tr,
            category.description_tr,
            " ".join(category.keywords_tr),
            " ".join(category.example_phrases_tr),
            " ".join(category.negative_keywords_tr),
        ]
        return " ".join(chunks).strip().lower()

