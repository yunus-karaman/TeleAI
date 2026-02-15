from __future__ import annotations

from solution_steps.generator import generate_kb_and_links_for_steps, generate_solution_steps_for_category
from solution_steps.quality import validate_solution_quality
from taxonomy.schema import load_taxonomy_file


def _generate_hashes() -> dict[str, str]:
    taxonomy = load_taxonomy_file("taxonomy/taxonomy.yaml")
    taxonomy_map = {category.category_id: category for category in taxonomy.categories}
    steps = generate_solution_steps_for_category(
        category_id="ROAMING_INTERNATIONAL",
        category_pattern={
            "top_symptoms": ["roaming internet yok"],
            "top_context_terms": ["mobil veri"],
            "top_trigger_terms": ["yurt disi girisi sonrasi"],
        },
        taxonomy_map=taxonomy_map,
        version="solution-steps-v1",
    )
    kb_items, links = generate_kb_and_links_for_steps(steps=steps, version="solution-steps-v1")
    quality = validate_solution_quality(
        steps=steps,
        kb_items=kb_items,
        links=links,
        target_categories=["ROAMING_INTERNATIONAL"],
        config={
            "min_steps_per_category": 6,
            "max_steps_per_category": 12,
            "min_level_counts": {"L1": 3, "L2": 2, "L3": 1},
        },
    )
    return quality["hashes"]


def test_solution_generation_is_deterministic() -> None:
    first = _generate_hashes()
    second = _generate_hashes()
    assert first == second

