from __future__ import annotations

import logging
from typing import Any, Protocol


class TaxonomyNormalizer(Protocol):
    def run_taxonomy_stage(self, *, config: dict[str, Any], mode: str, logger: logging.Logger) -> dict[str, Any]: ...
