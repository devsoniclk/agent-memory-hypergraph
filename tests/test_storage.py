import os
import tempfile
import pytest
from agent_hypergraph.models import Hypergraph, RelationType
from agent_hypergraph.storage import export_to_dot, export_to_mermaid, load_from_json, save_to_json


def test_atomic_save_and_load():
    hg = Hypergraph("persisted_test")
    hg.add_hyperedge("Rel", tail={"node1", "node2"}, head={"node3"}, edge_id="e1")

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "nested", "graph.json")
        save_to_json(hg, path)
        assert os.path.exists(path)

        loaded = load_from_json(path)
        assert loaded.name == "persisted_test"
        assert len(loaded.vertices) == 3
        assert len(loaded.hyperedges) == 1
        assert "e1" in loaded.hyperedges


def test_mermaid_export():
    hg = Hypergraph()
    hg.add_hyperedge("Action", tail={"agent", "tool"}, head={"output"}, edge_id="e1", relation_type=RelationType.CAUSAL)
    mermaid_str = export_to_mermaid(hg)

    assert "flowchart TD" in mermaid_str
    assert "hub_e1" in mermaid_str
    assert "-->" in mermaid_str
    assert "classDef hyperhub" in mermaid_str


def test_dot_export():
    hg = Hypergraph()
    hg.add_hyperedge("Undirected", tail={"x", "y", "z"}, edge_id="e_undir")
    dot_str = export_to_dot(hg)

    assert "digraph Hypergraph" in dot_str
    assert "hub_e_undir" in dot_str
    assert '"x"' in dot_str
