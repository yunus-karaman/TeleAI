from __future__ import annotations

import logging
from typing import Any, Protocol, Sequence


class Evaluator(Protocol):
    def evaluate(self, predictions: Sequence[Any], config: dict[str, Any]) -> Any: ...


class EvalStage(Protocol):
    def run_eval_stage(self, *, config: dict[str, Any], mode: str, logger: logging.Logger) -> dict[str, Any]: ...
