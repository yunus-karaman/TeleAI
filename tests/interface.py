from __future__ import annotations

from typing import Protocol


class TestSuite(Protocol):
    def run(self) -> int: ...
