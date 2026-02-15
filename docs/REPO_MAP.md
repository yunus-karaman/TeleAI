# Repo Map

## Repository Root
- Entry point: `main.py`
- Primary config: `config.yaml`
- Artifacts root: `artifacts/`
- Runtime logs: `artifacts/logs/pipeline.jsonl`

## Stage Flow (`main.py`)
- `preprocess` -> `preprocess/pipeline.py::run_preprocess_stage`
- `taxonomy` -> `taxonomy/pipeline.py::run_taxonomy_stage`
- `solution_steps` -> `solution_steps/pipeline.py::run_solution_steps_stage`
- `graph` -> `graph/pipeline.py::run_graph_stage`
- `train_llm` -> `training/pipeline.py::run_train_llm_stage`
- `infer` -> `models/infer.py::run_infer_stage`
- `eval` -> `evaluation/pipeline.py::run_eval_stage`

## Core Modules
- `preprocess/`: temizleme, PII maskleme, duplicate tespiti
- `taxonomy/`: kategori atama (kural + embedding hibrit), split üretimi
- `solution_steps/`: adım/KB/link üretimi, linting ve kalite kontrolleri
- `graph/`: graph build, embedding cache, retrieval ve retrieval eval
- `models/`: constrained inference, safety, template validation
- `training/`: dataset build, LoRA training orchestration (bu çalışmada training yok)
- `evaluation/`: hallucination/security/pii/task metric değerlendirmesi
- `api/`: FastAPI chat servis katmanı
- `scripts/`: config/logging/reproducibility yardımcıları
- `debug.py`: tek komutta sistem sağlık kontrolü (`--check all`)
- `scripts/integrate_solution_dataset.py`: çözüm dataset zip entegrasyonu + kanonik şema normalizasyonu

## Canonical Data/Artifact Paths (`config.yaml`)
- Raw dataset: `paths.dataset` (`sikayetler.jsonl`)
- Clean complaints: `paths.clean_complaints`
- Labeled complaints: `paths.labeled_complaints`
- Taxonomy file: `taxonomy.taxonomy_file` (canonical: `taxonomy/taxonomy.yaml`)
- Solution steps: `paths.solution_steps_jsonl`
- KB: `paths.kb_jsonl`
- Step-KB links: `paths.step_kb_links_jsonl`
- Graph nodes/edges/stats: `paths.graph_nodes`, `paths.graph_edges`, `paths.graph_stats`
- Eval outputs: `paths.eval_dir` altındaki raporlar

## Runtime Controls
- Determinism: `reproducibility.seed`, `reproducibility.deterministic`
- Mode profile merge: `scripts/config_loader.py`
- Mode-specific overrides: `mode_profiles.SMOKE`, `mode_profiles.FULL`
- Evaluation safety gates: `evaluation.safety_gates`

## Key Observations (Current Baseline)
- FULL mode mock training fallback kapatıldı ve config invariant olarak doğrulanıyor.
- Çözüm dataset entegrasyonu kanonik dosyalara normalize edilerek yazılıyor (`taxonomy/` + `artifacts/`).
- Solution dataset integrity kontrolü `graph` ve `train_llm` retrieval kaynak hazırlığında precheck olarak zorunlu.
- Merkezi fail-fast gate mekanizması aktif: FULL -> `artifacts/aborted_reason.json`, SMOKE -> `artifacts/smoke_notice.json`.
