# Turkish Telecom Complaint Assistant: Foundation Architecture

## 1) System Goal
Build a deterministic, evidence-constrained Turkish complaint assistant that supports:
- Graph-based reasoning (Graph-RAG; optional GNN later)
- Lightweight 7B-8B Turkish-capable LLM
- LoRA/QLoRA fine-tuning on H100
- Live agent escalation
- Strict hallucination control and reproducibility

This phase only establishes the production-ready foundation and contracts.

## 2) End-to-End Pipeline
1. Dataset schema analysis (`data/schema_analysis.py`)
2. Strict ingestion + quarantine (`data/ingestion.py`)
3. Preprocessing (stub)
4. Taxonomy normalization (stub)
5. Solution step generation (stub)
6. KB structuring (stub)
7. Graph construction (stub)
8. Retrieval + evidence packing (stub)
9. Training orchestration (stub)
10. Evaluation report persistence

Entrypoints:
- `python main.py --mode SMOKE`
- `python main.py --mode FULL`

## 3) Mode Behavior
### SMOKE
- Full stage traversal on subset (`mode_runtime.sample_size`)
- H100 target <= 60 minutes
- Minimal training profile (`epochs=1`, low max steps)

### FULL
- Full dataset
- Full training profile
- Checkpointing and resume in config (`training.checkpoint_dir`, `training.resume_from_checkpoint`)

## 4) Hallucination and Evidence Guardrails
- `hallucination_control.require_evidence=true`
- `max_unsupported_claims=0`
- Evidence coverage threshold (`citation_coverage_min`)
- Abstain/guard thresholds before response emission
- Escalation triggers for low confidence or critical sentiment

## 5) Reproducibility and Reliability
- Global seeding (`random`, `numpy`, `torch`)
- Deterministic flags (`torch.use_deterministic_algorithms`)
- JSON structured logs for every stage
- Invalid records quarantined in `artifacts/quarantine.jsonl`
- Fail-fast stage semantics with controlled process exit

## 6) Data Contracting
Strict Pydantic contracts are implemented in `data/schemas.py` for:
- `RawComplaint`
- `CleanComplaint`
- `NormalizedComplaint`
- `SolutionStep`
- `KBParagraph`
- `GraphNode`
- `GraphEdge`
- `TrainingExample`
- `EvidencePack`
- `EvaluationReport`

All schemas:
- Forbid unknown fields
- Use strict types
- Include version metadata (`schema_name`, `schema_version`, `schema_revision`)

## 7) Integration Strategy
- Unit tests by module category
- Integration smoke test for `main.py --mode SMOKE`
- Health test validating resilience to malformed records and missing fields

