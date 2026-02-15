from __future__ import annotations

import pytest
from pydantic import ValidationError

from training.lora_trainer import LoRAConfigModel, TrainerConfigModel


def test_lora_config_accepts_valid_values() -> None:
    cfg = LoRAConfigModel.model_validate(
        {
            "r": 64,
            "alpha": 16,
            "dropout": 0.05,
            "target_modules": ["q_proj", "k_proj", "v_proj"],
            "use_qlora_4bit": True,
        }
    )
    assert cfg.r == 64


def test_lora_config_rejects_invalid_rank() -> None:
    with pytest.raises(ValidationError):
        LoRAConfigModel.model_validate(
            {
                "r": 0,
                "alpha": 16,
                "dropout": 0.05,
                "target_modules": ["q_proj"],
                "use_qlora_4bit": True,
            }
        )


def test_trainer_config_rejects_invalid_batch_size() -> None:
    with pytest.raises(ValidationError):
        TrainerConfigModel.model_validate(
            {
                "learning_rate": 2.0e-4,
                "weight_decay": 0.01,
                "warmup_ratio": 0.03,
                "gradient_accumulation_steps": 4,
                "max_seq_len": 1536,
                "num_epochs": 1,
                "batch_size": 0,
                "eval_steps": 50,
                "save_steps": 50,
                "logging_steps": 10,
                "max_oom_retries": 2,
                "auto_batch_reduce_on_oom": True,
                "early_stopping_patience": 2,
            }
        )
