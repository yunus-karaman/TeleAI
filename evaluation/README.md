# evaluation

## Responsibility
Offline/online metrics, hallucination audits, adversarial security tests, KVKK/PII leakage checks, and safety-gated reporting.

## Run
- `python main.py --stage eval --mode SMOKE`
- `python main.py --stage eval --mode FULL`

## Outputs
- `artifacts/eval/hallucination_report.json`
- `artifacts/eval/hallucination_report.md`
- `artifacts/eval/security_adversarial_report.json`
- `artifacts/eval/security_adversarial_report.md`
- `artifacts/eval/pii_leak_report.json`
- `artifacts/eval/pii_leak_report.md`
- `artifacts/eval/task_metrics_report.json`
- `artifacts/eval/task_metrics_report.md`
- `artifacts/eval/combined_dashboard.json`

## Interface
Contract definitions live in interface.py.
