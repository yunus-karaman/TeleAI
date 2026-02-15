from __future__ import annotations

from taxonomy.assignment import HybridTaxonomyAssigner
from taxonomy.schema import load_taxonomy_file


def test_hybrid_assignment_deterministic() -> None:
    taxonomy = load_taxonomy_file("taxonomy/taxonomy.yaml")
    assigner = HybridTaxonomyAssigner(
        taxonomy=taxonomy,
        config={
            "min_confidence": 0.55,
            "low_confidence_policy": "other",
            "review_margin_threshold": 0.08,
            "rule_weight": 0.55,
            "embedding_weight": 0.45,
            "keyword_weight": 1.0,
            "negative_weight": 0.8,
            "example_weight": 1.2,
            "embedding": {
                "max_features": 3000,
                "ngram_min": 1,
                "ngram_max": 2,
                "min_df": 1,
            },
        },
        seed=42,
    )
    texts = [
        "Faturam odendigim halde borc gorunuyor ve odeme yansimiyor.",
        "Yurt disinda roaming paketi aldim ama internet calismiyor.",
        "Numara tasima basvurum reddedildi.",
    ]
    assigner.fit(texts)
    first = [assigner.assign(text).normalized_category for text in texts]
    second = [assigner.assign(text).normalized_category for text in texts]
    assert first == second

