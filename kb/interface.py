from __future__ import annotations

from typing import Any, Protocol, Sequence


class KnowledgeBaseBuilder(Protocol):
    def build(self, source_docs: Sequence[Any], config: dict[str, Any]) -> list[Any]: ...
