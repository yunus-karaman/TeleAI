# graph

## Responsibility
Graph construction, embedding caching, Graph-RAG retrieval, optional lightweight GNN message passing, and retrieval evaluation.

## Run
- `python main.py --stage graph --mode SMOKE`
- `python main.py --stage graph --mode FULL`

## Outputs
- `artifacts/graph/nodes.jsonl`
- `artifacts/graph/edges.jsonl`
- `artifacts/graph/graph_stats.json`
- `artifacts/retrieval_eval.json`
- `artifacts/retrieval_eval.md`
- `artifacts/review_pack_for_humans.jsonl`
- `artifacts/graph/gnn_embeddings.npz` (when `graph.use_gnn=true`)

## Interface
Contract definitions live in `interface.py`.
