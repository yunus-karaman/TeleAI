from __future__ import annotations

from typing import Any, Protocol


class UISession(Protocol):
    def render(self, state: dict[str, Any]) -> str: ...
