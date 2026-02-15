from __future__ import annotations

from taxonomy.schema import load_taxonomy_file


def test_taxonomy_yaml_validates() -> None:
    taxonomy = load_taxonomy_file("taxonomy/taxonomy.yaml")
    assert taxonomy.taxonomy_version == "1.0.0"
    assert 15 <= len(taxonomy.categories) <= 30
    category_ids = [category.category_id for category in taxonomy.categories]
    assert "OTHER" in category_ids
    assert len(category_ids) == len(set(category_ids))

