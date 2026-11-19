import io
import json
import time
import pytest
from agent_hypergraph.conflict import ConflictDetector, EpistemicConflict
from agent_hypergraph.mcp_server import AgentHypergraphMCPServer
from agent_hypergraph.models import Hypergraph, RelationType, VertexType
from agent_hypergraph.retrieval import HypergraphRetriever
from agent_hypergraph.spectral import (
    compute_hypergraph_laplacian,
    compute_hypergraph_pagerank,
    hypergraph_random_walk,
)
from agent_hypergraph.temporal import DecayModel, TemporalConsolidator


def test_pagerank_personalization():
    hg = Hypergraph()
    hg.add_hyperedge("e1", tail={"v1", "v2"}, head={"v3"})
    hg.add_hyperedge("e2", tail={"v3", "v4"}, head={"v5"})

    # Strongly personalize on v5
    ranks = compute_hypergraph_pagerank(hg, personalization={"v5": 10.0, "v1": 0.1})
    assert ranks["v5"] > ranks["v1"]


def test_vertex_update_and_properties():
    hg = Hypergraph()
    v = hg.add_vertex("Initial Label", vertex_type=VertexType.TOOL, vertex_id="tool1", properties={"version": 1})
    assert v.properties["version"] == 1

    # Update vertex
    v2 = hg.add_vertex("Updated Label", vertex_type=VertexType.TOOL, vertex_id="tool1", properties={"version": 2, "author": "nova"})
    assert v2.id == "tool1"
    assert v2.label == "Updated Label"
    assert v2.properties["version"] == 2
    assert v2.properties["author"] == "nova"


def test_touch_and_access_tracking():
    hg = Hypergraph()
    e = hg.add_hyperedge("edge", tail={"a"}, head={"b"}, edge_id="e1")
    t0 = e.last_accessed_at

    import time
    time.sleep(0.01)
    hg.touch(["e1"])
    assert e.last_accessed_at >= t0


def test_protected_edges_consolidation():
    hg = Hypergraph()
    e = hg.add_hyperedge("vital", tail={"root"}, edge_id="e_vital", weight=0.01, metadata={"permanent": True})
    tc = TemporalConsolidator(prune_threshold=0.1)

    res = tc.consolidate(hg, current_time=time.time() + 10000)
    assert "e_vital" not in res["pruned_edge_ids"]
    assert "e_vital" in hg.hyperedges


def test_multiple_conflicts_detection():
    hg = Hypergraph()
    # Value conflict
    hg.add_hyperedge("assert_A", tail={"entity1", "param"}, metadata={"value": "100"}, edge_id="ea")
    hg.add_hyperedge("assert_B", tail={"entity1", "param"}, metadata={"value": "200"}, edge_id="eb")

    # Unrelated edge
    hg.add_hyperedge("unrelated", tail={"other1", "other2"}, edge_id="e_other")

    detector = ConflictDetector(jaccard_threshold=0.5)
    conflicts = detector.detect_conflicts(hg)

    assert len(conflicts) == 1
    assert conflicts[0].edge_a_id == "ea"
    assert conflicts[0].edge_b_id == "eb"
    assert "Incompatible assigned values" in conflicts[0].description


def test_causal_path_self_target():
    hg = Hypergraph()
    retriever = HypergraphRetriever()
    res = retriever.find_causal_chain(hg, premises={"nodeA", "nodeB"}, target="nodeA")
    assert res.found is True
    assert res.visited_vertices == ["nodeA"]


def test_stdio_mcp_run_loop(monkeypatch):
    server = AgentHypergraphMCPServer()
    input_lines = (
        json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize"}) + "\n"
        + json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "add_vertex", "arguments": {"label": "LoopNode"}}}) + "\n"
    )

    monkeypatch.setattr("sys.stdin", io.StringIO(input_lines))
    out_capture = io.StringIO()
    monkeypatch.setattr("sys.stdout", out_capture)

    server.run_stdio()

    output = out_capture.getvalue().strip().split("\n")
    assert len(output) == 2
    r1 = json.loads(output[0])
    assert r1["id"] == 1
    r2 = json.loads(output[1])
    assert r2["id"] == 2
    assert "LoopNode" in output[1]


def test_random_walk_invalid_start():
    hg = Hypergraph()
    walk = hypergraph_random_walk(hg, "nonexistent")
    assert walk == []


def test_disconnected_laplacian():
    hg = Hypergraph()
    hg.add_vertex("island1", vertex_id="i1")
    hg.add_vertex("island2", vertex_id="i2")
    lap = compute_hypergraph_laplacian(hg)
    assert lap["num_vertices"] == 2
    assert lap["num_hyperedges"] == 0
    assert lap["vertex_degrees"]["i1"] == 0.0


def test_linear_decay_boundary():
    tc = TemporalConsolidator(decay_model=DecayModel.LINEAR, half_life_seconds=10.0)
    w = tc.calculate_decay(original_weight=1.0, decay_rate=0.05, elapsed_seconds=30.0)
    assert w == 0.0
