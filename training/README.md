# training

## Responsibility
LoRA/QLoRA training orchestration, deterministic SFT dataset creation, constrained quick evaluation, and checkpoint/run management.

## Run
- `python main.py --stage train_llm --mode SMOKE`
- `python main.py --stage train_llm --mode FULL`

## Outputs
- `artifacts/training/task2_sft_train.jsonl`
- `artifacts/training/task2_sft_val.jsonl`
- `artifacts/models/<run_id>/...`
- `artifacts/training_eval_quick.json`
- `artifacts/training_eval_quick.md`

## Notes
- Task-2 dataset is rendered deterministically from Graph-RAG `EvidencePack` outputs.
- If HF/PEFT stack or model weights are unavailable, mock fallback can keep the stage non-crashing (`training_llm.trainer.fallback_to_mock_on_failure`).
- Offline mode is supported via `model.local_files_only=true` after model download.

## Interface
Contract definitions live in interface.py.
