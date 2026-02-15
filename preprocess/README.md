# preprocess

## Responsibility
KVKK-safe complaint preprocessing:
- strict raw-schema validation with quarantine
- text cleaning and script/ad noise removal
- multi-complaint detection + primary extraction/split policy
- layered Turkish PII masking
- length normalization
- near-duplicate detection and mode-specific retention policy

## Run
- `python main.py --stage preprocess --mode SMOKE`
- `python main.py --stage preprocess --mode FULL`

## Outputs
- `artifacts/complaints_clean.jsonl`
- `artifacts/preprocess_report.json`
- `artifacts/duplicates_report.json`
- `artifacts/quarantine.jsonl`

## Interface
Contract definitions live in `interface.py`.
