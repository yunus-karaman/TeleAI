# taxonomy

## Responsibility
Stable telecom taxonomy management, deterministic category assignment, leakage-safe splitting, and baseline evaluation.

## Run
- `python main.py --stage taxonomy --mode SMOKE`
- `python main.py --stage taxonomy --mode FULL`

## Outputs
- `taxonomy/taxonomy.yaml`
- `artifacts/complaints_labeled.jsonl`
- `artifacts/splits/train.jsonl`
- `artifacts/splits/val.jsonl`
- `artifacts/splits/test.jsonl`
- `artifacts/splits/hard_test.jsonl`
- `artifacts/taxonomy_report.json`
- `artifacts/taxonomy_report.md`

## Interface
Contract definitions live in `interface.py`.
