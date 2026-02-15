from __future__ import annotations

from data.schemas import GraphEdge, GraphNode


def test_graph_node_and_edge_schema_validation() -> None:
    node = GraphNode(
        node_id="c-001",
        node_type="complaint",
        label="c-001",
        attributes={"normalized_category": "BILLING_PAYMENTS", "category_confidence": 0.81},
        source_ids=[],
        confidence=0.81,
    )
    edge = GraphEdge(
        edge_id="EDGE.CLASSIFIED_AS.0001",
        source_node_id="c-001",
        target_node_id="BILLING_PAYMENTS",
        relation_type="CLASSIFIED_AS",
        weight=0.81,
        evidence_ids=[],
        bidirectional=False,
    )
    assert node.node_type == "complaint"
    assert edge.relation_type == "CLASSIFIED_AS"

