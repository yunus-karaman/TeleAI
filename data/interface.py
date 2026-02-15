from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol


class SchemaAnalyzer(Protocol):
    def analyze(self, dataset_path: Path, output_path: Path) -> dict[str, Any]: ...


class ComplaintLoader(Protocol):
    def load(self, dataset_path: Path, sample_size: int | None = None) -> tuple[list[Any], dict[str, int]]: ...
