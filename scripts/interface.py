from __future__ import annotations

from typing import Protocol


class ScriptTask(Protocol):
    def run(self) -> int: ...
