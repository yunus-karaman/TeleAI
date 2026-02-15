from __future__ import annotations

from typing import Any, Protocol, Sequence


class EvidenceRetriever(Protocol):
    def retrieve(self, query_items: Sequence[Any], config: dict[str, Any]) -> list[Any]: ...
