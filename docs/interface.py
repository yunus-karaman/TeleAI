from __future__ import annotations

from typing import Protocol


class DocumentationSet(Protocol):
    def publish(self) -> None: ...
