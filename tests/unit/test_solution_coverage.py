from __future__ import annotations

from solution_steps.generator import generate_kb_and_links_for_steps, generate_solution_steps_for_category
from solution_steps.quality import validate_solution_quality
from taxonomy.schema import load_taxonomy_file


def test_solution_quality_coverage_checks_pass_for_generated_category() -> None:
    taxonomy = load_taxonomy_file("taxonomy/taxonomy.yaml")
    taxonomy_map = {category.category_id: category for category in taxonomy.categories}
    steps = generate_solution_steps_for_category(
        category_id="BILLING_PAYMENTS",
        category_pattern={
            "top_symptoms": ["fatura yuksek"],
            "top_context_terms": ["uygulama"],
            "top_trigger_terms": ["odeme sonrasi"],
        },
        taxonomy_map=taxonomy_map,
        version="solution-steps-v1",
    )
    kb_items, links = generate_kb_and_links_for_steps(steps=steps, version="solution-steps-v1")

    quality = validate_solution_quality(
        steps=steps,
        kb_items=kb_items,
        links=links,
        target_categories=["BILLING_PAYMENTS"],
        config={
            "min_steps_per_category": 6,
            "max_steps_per_category": 12,
            "min_level_counts": {"L1": 3, "L2": 2, "L3": 1},
        },
    )
    assert quality["missing_evidence_count"] == 0
    assert quality["errors"] == []
    assert quality["count_per_category"]["BILLING_PAYMENTS"] == 6

