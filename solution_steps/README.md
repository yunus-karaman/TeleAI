# solution_steps

## Responsibility
Category pattern mining, deterministic solution step generation, KB evidence generation, safety linting, and step-evidence linking.

## Run
- `python main.py --stage solution_steps --mode SMOKE`
- `python main.py --stage solution_steps --mode FULL`

## Outputs
- `artifacts/category_patterns.json`
- `artifacts/solution_steps.jsonl`
- `artifacts/kb.jsonl`
- `artifacts/step_kb_links.jsonl`
- `artifacts/solution_steps_summary.json`
- `artifacts/solution_step_lint_report.json`
- `artifacts/kb_lint_report.json`

## Interface
Contract definitions live in `interface.py`.
