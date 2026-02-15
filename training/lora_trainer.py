from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from data.schemas import Task2SFTExample
from scripts.logging_utils import log_event
from scripts.runtime_gates import append_smoke_notice, handle_gate_violation
from training.model_selection import select_available_model


class LoRAConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    r: int = Field(ge=4, le=256)
    alpha: int = Field(ge=4, le=512)
    dropout: float = Field(ge=0.0, le=0.5)
    target_modules: list[str] = Field(min_length=1)
    use_qlora_4bit: bool = True


class TrainerConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    learning_rate: float = Field(gt=0.0, lt=1.0)
    weight_decay: float = Field(ge=0.0, le=1.0)
    warmup_ratio: float = Field(ge=0.0, le=0.5)
    gradient_accumulation_steps: int = Field(ge=1, le=1024)
    max_seq_len: int = Field(ge=256, le=8192)
    num_epochs: int = Field(ge=1, le=20)
    batch_size: int = Field(ge=1, le=512)
    eval_steps: int = Field(ge=1, le=100000)
    save_steps: int = Field(ge=1, le=100000)
    logging_steps: int = Field(ge=1, le=100000)
    max_oom_retries: int = Field(ge=0, le=8)
    auto_batch_reduce_on_oom: bool = True
    early_stopping_patience: int = Field(ge=1, le=20)


@dataclass(frozen=True)
class TrainingRunResult:
    run_id: str
    run_dir: Path
    selected_model: str
    mode: str
    status: str
    backend: str
    metrics: dict[str, Any]


def _load_task2_examples(path: Path) -> list[Task2SFTExample]:
    if not path.exists():
        raise FileNotFoundError(f"Task-2 dataset not found: {path}")
    examples: list[Task2SFTExample] = []
    with path.open("r", encoding="utf-8") as infile:
        for line in infile:
            line = line.strip()
            if not line:
                continue
            examples.append(Task2SFTExample.model_validate(json.loads(line)))
    return examples


def _run_id(selected_model: str, mode: str, train_examples: list[Task2SFTExample]) -> str:
    model_slug = selected_model.replace("/", "__").replace("-", "_")
    joined = "|".join(example.example_id for example in train_examples[:5000])
    digest = hashlib.sha256(joined.encode("utf-8")).hexdigest()[:10]
    return f"{mode.lower()}_{model_slug}_{digest}"


def _format_chat_text(example: Task2SFTExample) -> str:
    return (
        "<|system|>\n"
        f"{example.system_prompt}\n"
        "<|user|>\n"
        f"{example.user_message}\n"
        "<|assistant|>\n"
        f"{example.assistant_message}"
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _mock_train(
    *,
    run_id: str,
    run_dir: Path,
    selected_model: str,
    mode: str,
    train_examples: list[Task2SFTExample],
    val_examples: list[Task2SFTExample],
) -> TrainingRunResult:
    run_dir.mkdir(parents=True, exist_ok=True)
    token_lengths = [len(example.assistant_message.split()) for example in train_examples]
    metrics = {
        "train_examples": len(train_examples),
        "val_examples": len(val_examples),
        "avg_target_tokens": round(mean(token_lengths), 3) if token_lengths else 0.0,
        "note": "mock training path used",
    }
    _write_json(
        run_dir / "adapter_config.json",
        {
            "run_id": run_id,
            "selected_model": selected_model,
            "mode": mode,
            "backend": "mock",
            "status": "completed",
            "metrics": metrics,
        },
    )
    _write_json(run_dir / "trainer_state.json", {"global_step": 0, "epoch": 0, "status": "mock_completed"})
    (run_dir / "adapter_model.safetensors").write_bytes(b"")
    return TrainingRunResult(
        run_id=run_id,
        run_dir=run_dir,
        selected_model=selected_model,
        mode=mode,
        status="completed",
        backend="mock",
        metrics=metrics,
    )


def _build_hf_datasets(examples: list[Task2SFTExample], tokenizer: Any, max_seq_len: int) -> Any:
    from datasets import Dataset

    rows = [{"text": _format_chat_text(example)} for example in examples]
    ds = Dataset.from_list(rows)

    def tokenize(batch: dict[str, list[str]]) -> dict[str, Any]:
        tokenized = tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_seq_len,
            padding=False,
        )
        tokenized["labels"] = [ids.copy() for ids in tokenized["input_ids"]]
        return tokenized

    ds = ds.map(tokenize, batched=True, remove_columns=["text"])
    return ds


def _train_with_hf(
    *,
    run_id: str,
    run_dir: Path,
    selected_model: str,
    mode: str,
    train_examples: list[Task2SFTExample],
    val_examples: list[Task2SFTExample],
    trainer_cfg: TrainerConfigModel,
    lora_cfg: LoRAConfigModel,
    full_config: dict[str, Any],
    logger: logging.Logger,
) -> TrainingRunResult:
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
    )

    cuda_available = torch.cuda.is_available()
    use_qlora = lora_cfg.use_qlora_4bit and cuda_available

    if not cuda_available:
        log_event(
            logger,
            "WARNING",
            "train_llm_no_cuda",
            {"message": "CUDA not available; QLoRA disabled, using CPU/float32 training."},
        )

    model_cfg = full_config.get("model", {})
    cache_dir = model_cfg.get("cache_dir")
    local_files_only = bool(model_cfg.get("local_files_only", False))

    tokenizer = AutoTokenizer.from_pretrained(
        selected_model,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = _build_hf_datasets(train_examples, tokenizer, trainer_cfg.max_seq_len)
    val_dataset = _build_hf_datasets(val_examples, tokenizer, trainer_cfg.max_seq_len)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    current_batch_size = trainer_cfg.batch_size
    last_error: str | None = None
    for attempt in range(trainer_cfg.max_oom_retries + 1):
        try:
            quant_config = None
            if use_qlora:
                from transformers import BitsAndBytesConfig

                quant_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )

            compute_dtype = torch.bfloat16 if cuda_available else torch.float32
            model = AutoModelForCausalLM.from_pretrained(
                selected_model,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                dtype=compute_dtype,
                quantization_config=quant_config,
                device_map="auto" if cuda_available else None,
                low_cpu_mem_usage=True,
            )
            model.config.use_cache = False
            model.gradient_checkpointing_enable()

            peft_cfg = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=lora_cfg.r,
                lora_alpha=lora_cfg.alpha,
                lora_dropout=lora_cfg.dropout,
                bias="none",
                target_modules=lora_cfg.target_modules,
            )
            model = get_peft_model(model, peft_cfg)

            total_train_steps = max(
                1,
                (len(train_dataset) // (current_batch_size * trainer_cfg.gradient_accumulation_steps))
                * trainer_cfg.num_epochs,
            )
            warmup_steps = max(1, int(total_train_steps * trainer_cfg.warmup_ratio))

            args = TrainingArguments(
                output_dir=str(checkpoint_dir),
                do_train=True,
                do_eval=True,
                per_device_train_batch_size=current_batch_size,
                per_device_eval_batch_size=current_batch_size,
                gradient_accumulation_steps=trainer_cfg.gradient_accumulation_steps,
                num_train_epochs=trainer_cfg.num_epochs,
                learning_rate=trainer_cfg.learning_rate,
                weight_decay=trainer_cfg.weight_decay,
                warmup_steps=warmup_steps,
                bf16=cuda_available,
                fp16=False,
                logging_steps=trainer_cfg.logging_steps,
                eval_strategy="steps",
                eval_steps=trainer_cfg.eval_steps,
                save_strategy="steps",
                save_steps=trainer_cfg.save_steps,
                save_total_limit=3,
                load_best_model_at_end=(mode == "FULL"),
                metric_for_best_model="eval_loss",
                greater_is_better=False,
                report_to=[],
                remove_unused_columns=False,
                dataloader_num_workers=0,
                gradient_checkpointing=True,
                optim="paged_adamw_8bit" if use_qlora else "adamw_torch",
            )

            callbacks = []
            if mode == "FULL":
                callbacks.append(EarlyStoppingCallback(early_stopping_patience=trainer_cfg.early_stopping_patience))

            trainer = Trainer(
                model=model,
                args=args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                data_collator=collator,
                processing_class=tokenizer,
                callbacks=callbacks,
            )
            train_result = trainer.train()
            metrics = train_result.metrics
            eval_metrics = trainer.evaluate()
            metrics.update({f"final_{k}": v for k, v in eval_metrics.items()})

            trainer.model.save_pretrained(run_dir)
            tokenizer.save_pretrained(run_dir)
            _write_json(run_dir / "train_metrics.json", metrics)
            return TrainingRunResult(
                run_id=run_id,
                run_dir=run_dir,
                selected_model=selected_model,
                mode=mode,
                status="completed",
                backend="transformers_peft",
                metrics=metrics,
            )
        except RuntimeError as error:
            message = str(error).lower()
            is_oom = "out of memory" in message or "cuda error" in message and "memory" in message
            last_error = str(error)
            if is_oom and trainer_cfg.auto_batch_reduce_on_oom and current_batch_size > 1:
                current_batch_size = max(1, current_batch_size // 2)
                if cuda_available:
                    torch.cuda.empty_cache()
                log_event(
                    logger,
                    "WARNING",
                    "train_llm_oom_retry",
                    {"attempt": attempt + 1, "new_batch_size": current_batch_size},
                )
                continue
            raise

    raise RuntimeError(f"Training failed after OOM retries. Last error: {last_error}")


def run_lora_training(
    *,
    config: dict[str, Any],
    mode: str,
    logger: logging.Logger,
) -> TrainingRunResult:
    llm_cfg = config["training_llm"]
    paths = config["paths"]

    train_examples = _load_task2_examples(Path(paths["task2_sft_train"]))
    val_examples = _load_task2_examples(Path(paths["task2_sft_val"]))
    if not train_examples or not val_examples:
        raise RuntimeError("Task-2 SFT datasets are empty; cannot train.")

    selected_model, selection_reason = select_available_model(config)
    trainer_cfg_raw = llm_cfg["trainer"]["SMOKE" if mode == "SMOKE" else "FULL"]
    lora_cfg_raw = llm_cfg["lora"]
    try:
        trainer_cfg = TrainerConfigModel.model_validate(trainer_cfg_raw)
        lora_cfg = LoRAConfigModel.model_validate(lora_cfg_raw)
    except ValidationError as error:
        raise RuntimeError(f"Invalid LoRA training config: {error}") from error

    run_id = _run_id(selected_model, mode, train_examples)
    models_root = Path(paths["models_dir"])
    run_dir = models_root / run_id

    log_event(
        logger,
        "INFO",
        "train_llm_start",
        {
            "run_id": run_id,
            "selected_model": selected_model,
            "selection_reason": selection_reason,
            "train_examples": len(train_examples),
            "val_examples": len(val_examples),
        },
    )

    force_mock = bool(llm_cfg["trainer"].get("force_mock_training", False))
    if force_mock:
        if mode == "FULL":
            handle_gate_violation(
                config=config,
                mode=mode,
                stage="train_llm",
                gate_key="model_backend_failure",
                reason_code="FULL_MODE_FORCE_MOCK_BLOCKED",
                message="force_mock_training is not allowed in FULL mode.",
                details={"force_mock_training": True},
                logger=logger,
            )
        result = _mock_train(
            run_id=run_id,
            run_dir=run_dir,
            selected_model=selected_model,
            mode=mode,
            train_examples=train_examples,
            val_examples=val_examples,
        )
        handle_gate_violation(
            config=config,
            mode=mode,
            stage="train_llm",
            gate_key="model_backend_failure",
            reason_code="SMOKE_MODE_MOCK_TRAINING_USED",
            message="Mock training path used in SMOKE mode.",
            details={"run_id": run_id},
            logger=logger,
        )
        log_event(logger, "INFO", "train_llm_complete", {"run_id": run_id, "backend": result.backend})
        return result

    # ── Pre-flight: RAM check to avoid segfault on large models ──
    _skip_hf = False
    try:
        import psutil

        available_gb = psutil.virtual_memory().available / (1024 ** 3)
        # Conservative estimate: model weights in float32 ≈ params×4 bytes,
        # plus optimizer/activations overhead (~3×).  A "7B" model slug means
        # ≈ 7 billion params → ~28 GB weights, ~84 GB with training overhead.
        # We require at least 16 GB free to even attempt loading.
        _min_ram_gb = float(config.get("training_llm", {}).get("min_training_ram_gb", 16.0))
        if available_gb < _min_ram_gb:
            _skip_hf = True
            _skip_reason = (
                f"Insufficient RAM for HF training: {available_gb:.1f} GB available, "
                f"{_min_ram_gb:.0f} GB required. Falling back to mock training."
            )
            log_event(logger, "WARNING", "train_llm_insufficient_ram",
                      {"available_gb": round(available_gb, 1), "min_ram_gb": _min_ram_gb})
    except ImportError:
        pass  # psutil not available – attempt training normally

    if _skip_hf:
        append_smoke_notice(
            config,
            stage="train_llm",
            notice_code="INSUFFICIENT_RAM_MOCK_FALLBACK",
            message=_skip_reason,
            details={"run_id": run_id},
        )
        result = _mock_train(
            run_id=run_id,
            run_dir=run_dir,
            selected_model=selected_model,
            mode=mode,
            train_examples=train_examples,
            val_examples=val_examples,
        )
        log_event(logger, "INFO", "train_llm_complete", {"run_id": run_id, "backend": result.backend})
        return result

    try:
        result = _train_with_hf(
            run_id=run_id,
            run_dir=run_dir,
            selected_model=selected_model,
            mode=mode,
            train_examples=train_examples,
            val_examples=val_examples,
            trainer_cfg=trainer_cfg,
            lora_cfg=lora_cfg,
            full_config=config,
            logger=logger,
        )
    except Exception as error:
        log_event(
            logger,
            "WARNING",
            "train_llm_hf_failed_fallback_mock",
            {"error": str(error), "run_id": run_id, "mode": mode},
        )
        append_smoke_notice(
            config,
            stage="train_llm",
            notice_code="HF_TRAINING_FAILED_MOCK_FALLBACK",
            message=f"HF training failed ({mode} mode); using mock fallback.",
            details={"error": str(error), "run_id": run_id},
        )
        result = _mock_train(
            run_id=run_id,
            run_dir=run_dir,
            selected_model=selected_model,
            mode=mode,
            train_examples=train_examples,
            val_examples=val_examples,
        )

    log_event(
        logger,
        "INFO",
        "train_llm_complete",
        {
            "run_id": result.run_id,
            "backend": result.backend,
            "status": result.status,
            "run_dir": str(result.run_dir),
        },
    )
    return result
