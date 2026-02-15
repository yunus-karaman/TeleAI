from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, StrictStr, field_validator, model_validator


class TaxonomyCategory(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    category_id: StrictStr = Field(pattern=r"^[A-Z0-9_]+$")
    title_tr: StrictStr = Field(min_length=2)
    description_tr: StrictStr = Field(min_length=10)
    keywords_tr: list[StrictStr] = Field(min_length=1)
    negative_keywords_tr: list[StrictStr] = Field(default_factory=list)
    example_phrases_tr: list[StrictStr] = Field(min_length=1)
    escalation_default_unit: StrictStr | None = None
    risk_level_default: Literal["low", "medium", "high"]
    version: StrictStr = Field(min_length=1)

    @field_validator("keywords_tr", "negative_keywords_tr", "example_phrases_tr")
    @classmethod
    def normalize_lists(cls, values: list[StrictStr]) -> list[StrictStr]:
        normalized = sorted({value.strip() for value in values if value.strip()})
        if values is not None and len(values) > 0 and len(normalized) == 0:
            raise ValueError("List must contain at least one non-empty item.")
        return normalized


class TaxonomyFile(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    taxonomy_name: StrictStr
    taxonomy_version: StrictStr
    language: StrictStr = "tr"
    categories: list[TaxonomyCategory] = Field(min_length=15, max_length=30)

    @model_validator(mode="after")
    def validate_unique_ids(self) -> "TaxonomyFile":
        ids = [category.category_id for category in self.categories]
        if len(ids) != len(set(ids)):
            raise ValueError("category_id values must be unique")
        return self


def load_taxonomy_file(path: str | Path) -> TaxonomyFile:
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"taxonomy file not found: {target}")
    data = yaml.safe_load(target.read_text(encoding="utf-8"))
    return TaxonomyFile.model_validate(data)

