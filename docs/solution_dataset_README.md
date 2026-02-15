# Telecom Solution Dataset (Operator-agnostic, KVKK-safe)

This package contains a professional, structured dataset to power a telecom complaint assistant's **solution-step** and **evidence (KB)** layers.

## Contents
- `taxonomy.yaml` — 15 top-level telecom categories (operator-agnostic)
- `solution_steps.jsonl` — 150 SolutionStep records (10 per category; L1/L2/L3 mix)
- `kb.jsonl` — 300 KB paragraphs (2 per step)
- `step_kb_links.jsonl` — mapping from each step to its evidence paragraph IDs

## Design principles
- Turkish language
- KVKK-safe (no requests for TCKN/IBAN/IMEI/ICCID/full address/account numbers)
- Operator-agnostic best practices (no operator policy claims or refund guarantees)
- Evidence for every step (no evidence-less steps)
- Deterministic IDs:
  - Step: `STEP.<CATEGORY>.<NNN>`
  - KB: `KB.<CATEGORY>.<NNN>#P<k>`

## Schemas (high level)
### SolutionStep
- `step_id`, `category_id`, `level`, `title_tr`, `instructions_tr[]`, `required_inputs[]`,
  `success_check`, `stop_conditions`, `escalation_unit`, `risk_level`, `tags[]`, `version`

### KBParagraph
- `doc_id`, `paragraph_id`, `text_tr`, `applies_to_step_ids[]`, `source_type`, `confidence`, `version`

### Link
- `step_id`, `evidence_ids[]`, `rationale`, `version`

## Notes
This dataset is meant to be used as a **grounded library**:
- Retrieval selects steps from `solution_steps.jsonl`
- The assistant must cite KB evidence IDs from `kb.jsonl`
- If steps don't solve the issue, escalate using `escalation_unit`

Generated: 2026-02-15

## Canonicalization Note
- Integration normalizes records to project schema contracts.
- `solution_steps.jsonl` and `kb.jsonl` include `schema_name`, `schema_version`, `schema_revision`.
- `stop_conditions` is stored as a list for each step.
