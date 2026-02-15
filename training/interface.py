from __future__ import annotations

import logging
from typing import Any, Protocol, Sequence


class Trainer(Protocol):
    def train(self, examples: Sequence[Any], config: dict[str, Any]) -> dict[str, Any]: ...


class LLMTrainingStage(Protocol):
    def run_train_llm_stage(self, *, config: dict[str, Any], mode: str, logger: logging.Logger) -> dict[str, Any]: ...
