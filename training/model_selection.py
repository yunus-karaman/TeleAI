from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ModelCandidate:
    model_name: str
    family: str
    params_billion: float
    pros: str
    cons: str
    vram_estimate_h100_qlora_gb: float
    vram_estimate_h100_lora_bf16_gb: float
    license_note: str


def default_model_candidates() -> list[ModelCandidate]:
    return [
        ModelCandidate(
            model_name="Qwen/Qwen2.5-7B-Instruct",
            family="Qwen2.5",
            params_billion=7.0,
            pros="Strong multilingual quality including Turkish, robust instruction following.",
            cons="Tokenizer behavior differs from Llama family; prompt tuning may be needed.",
            vram_estimate_h100_qlora_gb=20.0,
            vram_estimate_h100_lora_bf16_gb=42.0,
            license_note="Research/demo usage should be verified against upstream model card.",
        ),
        ModelCandidate(
            model_name="mistralai/Mistral-7B-Instruct-v0.3",
            family="Mistral",
            params_billion=7.0,
            pros="Fast inference and mature ecosystem support.",
            cons="Turkish quality can be less consistent than Qwen in domain-heavy settings.",
            vram_estimate_h100_qlora_gb=18.0,
            vram_estimate_h100_lora_bf16_gb=39.0,
            license_note="Research/demo usage should be verified against upstream model card.",
        ),
    ]


def resolve_model_candidates(config: dict[str, Any]) -> list[ModelCandidate]:
    model_cfg = config.get("model", {})
    primary = str(model_cfg.get("base_model_name", "Qwen/Qwen2.5-7B-Instruct"))
    fallback = str(model_cfg.get("fallback_model_name", "mistralai/Mistral-7B-Instruct-v0.3"))

    candidate_map = {item.model_name: item for item in default_model_candidates()}
    ordered = [primary, fallback]
    resolved: list[ModelCandidate] = []
    for name in ordered:
        if name in candidate_map:
            resolved.append(candidate_map[name])
        else:
            resolved.append(
                ModelCandidate(
                    model_name=name,
                    family="custom",
                    params_billion=7.0,
                    pros="Configured by project.",
                    cons="No curated profile available.",
                    vram_estimate_h100_qlora_gb=22.0,
                    vram_estimate_h100_lora_bf16_gb=44.0,
                    license_note="Verify model license before training/deployment.",
                )
            )
    return resolved


def select_available_model(config: dict[str, Any]) -> tuple[str, str]:
    """
    Returns (selected_model_name, selection_reason).
    Preference order: configured primary, configured fallback.
    Availability check uses local cache when local_files_only=true.
    """
    model_cfg = config.get("model", {})
    local_files_only = bool(model_cfg.get("local_files_only", False))
    cache_dir = model_cfg.get("cache_dir")

    candidates = resolve_model_candidates(config)
    if local_files_only:
        try:
            from transformers import AutoConfig
        except ModuleNotFoundError:
            # Transformers may not be installed in lightweight environments.
            return candidates[0].model_name, "transformers_not_installed_using_primary"
        except ImportError:
            # Transformers may not be installed in lightweight environments.
            return candidates[0].model_name, "transformers_not_installed_using_primary"

        for candidate in candidates:
            try:
                AutoConfig.from_pretrained(
                    candidate.model_name,
                    local_files_only=True,
                    cache_dir=cache_dir,
                )
                return candidate.model_name, "local_cache_available"
            except OSError:
                continue
            except ValueError:
                continue
        return candidates[0].model_name, "local_cache_missing_using_primary"

    return candidates[0].model_name, "remote_or_cache_allowed_primary_selected"
