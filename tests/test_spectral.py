import pytest
from agent_hypergraph.models import Hypergraph, RelationType
from agent_hypergraph.spectral import (
    compute_hypergraph_laplacian,
    compute_hypergraph_pagerank,
    hypergraph_random_walk,
    hypergraph_transition_probability,
)


def create_sample_hypergraph() -> Hypergraph:
    hg = Hypergraph()
    # Edge 1 connects 1, 2, 3
    hg.add_hyperedge("e1", tail={"v1", "v2"}, head={"v3"}, edge_id="e1", weight=1.0)
    # Edge 2 connects 3, 4, 5
    hg.add_hyperedge("e2", tail={"v3"}, head={"v4", "v5"}, edge_id="e2", weight=2.0)
    # Edge 3 connects 5, 6
    hg.add_hyperedge("e3", tail={"v5"}, head={"v6"}, edge_id="e3", weight=1.0)
    return hg


def test_transition_probability():
    hg = create_sample_hypergraph()
    p_1_2 = hypergraph_transition_probability(hg, "v1", "v2")
    p_1_3 = hypergraph_transition_probability(hg, "v1", "v3")
    p_1_4 = hypergraph_transition_probability(hg, "v1", "v4")

    assert p_1_2 > 0.0
    assert p_1_3 > 0.0
    assert p_1_4 == 0.0  # Not in same hyperedge
    assert p_1_2 == pytest.approx(0.5)


def test_random_walk():
    hg = create_sample_hypergraph()
    walk = hypergraph_random_walk(hg, "v1", walk_length=15, restart_prob=0.1, seed=42)
    assert len(walk) == 15
    assert walk[0] == "v1"
    for step in walk:
        assert step in hg.vertices


def test_pagerank_computation():
    hg = create_sample_hypergraph()
    ranks = compute_hypergraph_pagerank(hg, damping=0.85, max_iter=100)
    assert len(ranks) == 6
    assert sum(ranks.values()) == pytest.approx(1.0, rel=1e-4)

    # v3 is the bridge between e1 and e2, should have high centrality
    assert ranks["v3"] > ranks["v1"]
    assert ranks["v3"] > ranks["v6"]


def test_pagerank_single_node():
    hg = Hypergraph()
    hg.add_vertex("solo", vertex_id="solo")
    ranks = compute_hypergraph_pagerank(hg)
    assert ranks == {"solo": 1.0}


def test_hypergraph_laplacian():
    hg = create_sample_hypergraph()
    lap = compute_hypergraph_laplacian(hg)
    assert lap["num_vertices"] == 6
    assert lap["num_hyperedges"] == 3
    assert lap["hyperedge_ranks"]["e1"] == 3
    assert lap["hyperedge_ranks"]["e2"] == 3
    assert lap["hyperedge_ranks"]["e3"] == 2
    assert lap["vertex_degrees"]["v3"] > lap["vertex_degrees"]["v1"]
