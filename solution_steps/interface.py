from __future__ import annotations

import logging
from typing import Any, Protocol


class SolutionStepGenerator(Protocol):
    def run_solution_steps_stage(self, *, config: dict[str, Any], mode: str, logger: logging.Logger) -> dict[str, Any]: ...
