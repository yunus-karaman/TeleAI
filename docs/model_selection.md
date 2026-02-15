# LLM Model Selection (Stage 6)

## Candidate Set (7B-8B)

### 1) `Qwen/Qwen2.5-7B-Instruct` (Primary)
- Turkish capability: Strong multilingual quality, generally reliable for Turkish instructions.
- Pros:
  - Good instruction-following and structured output behavior.
  - Stronger multilingual support for mixed-domain complaints.
- Cons:
  - Prompt formatting and tokenizer behavior may require task-specific tuning.
- Approximate H100 VRAM:
  - QLoRA 4-bit training: ~20 GB
  - LoRA BF16 full-weight loading: ~42 GB

### 2) `mistralai/Mistral-7B-Instruct-v0.3` (Fallback)
- Turkish capability: Good baseline, but can be less stable than Qwen for Turkish telecom phrasing.
- Pros:
  - Fast and widely supported inference/training stack.
  - Good latency/quality tradeoff for 7B.
- Cons:
  - Turkish edge-case robustness may be lower than Qwen.
- Approximate H100 VRAM:
  - QLoRA 4-bit training: ~18 GB
  - LoRA BF16 full-weight loading: ~39 GB

## Offline/Resilient Operation
- Config knobs:
  - `model.base_model_name`
  - `model.fallback_model_name`
  - `model.cache_dir`
  - `model.local_files_only`
- Recommended offline flow:
  1. Download model once in connected environment.
  2. Re-run with `model.local_files_only=true`.
  3. Keep `model.cache_dir` on persistent storage.

## Quantization Plan
- Training:
  - Primary path: QLoRA 4-bit (`nf4`) with PEFT adapters.
  - Fallback path: LoRA full precision if 4-bit stack unavailable.
- Inference:
  - Adapter + base model merge optional for deployment optimization.
  - Later optimization target: 4-bit inference or merged 8-bit export for lower-latency serving.
