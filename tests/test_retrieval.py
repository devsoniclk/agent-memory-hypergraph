import pytest
from agent_hypergraph.models import Hypergraph, RelationType
from agent_hypergraph.retrieval import HypergraphRetriever


def test_expand_subgraph():
    hg = Hypergraph()
    hg.add_hyperedge("e1", tail={"seed1"}, head={"intermediate1"}, edge_id="e1")
    hg.add_hyperedge("e2", tail={"intermediate1"}, head={"hop2_node"}, edge_id="e2")
    hg.add_hyperedge("e3", tail={"hop2_node"}, head={"hop3_node"}, edge_id="e3")

    retriever = HypergraphRetriever(default_max_hops=2)
    v_map, e_map = retriever.expand_subgraph(hg, seed_vertices=["seed1"], max_hops=2)

    assert "seed1" in v_map
    assert "intermediate1" in v_map
    assert "hop2_node" in v_map
    # hop3_node is 3 hops away, should not be included
    assert "hop3_node" not in v_map
    assert "e1" in e_map
    assert "e2" in e_map
    assert "e3" not in e_map


def test_causal_chain_success():
    hg = Hypergraph()
    # Step 1: {Observation} -> {Hypothesis}
    hg.add_hyperedge("Form hypothesis", tail={"obs"}, head={"hyp"}, edge_id="e1", relation_type=RelationType.CAUSAL)
    # Step 2: {Hypothesis, Tool} -> {Verification}
    hg.add_hyperedge("Test hypothesis", tail={"hyp", "tool"}, head={"verified"}, edge_id="e2", relation_type=RelationType.CAUSAL)

    retriever = HypergraphRetriever()
    res = retriever.find_causal_chain(hg, premises={"obs", "tool"}, target="verified")

    assert res.found is True
    assert len(res.path_edges) == 2
    assert "verified" in res.visited_vertices
    assert len(res.explanatory_gap) == 0


def test_causal_chain_missing_gap():
    hg = Hypergraph()
    # Target requires {hyp, secret_key} -> {access}
    hg.add_hyperedge("Secret action", tail={"hyp", "secret_key"}, head={"access"}, edge_id="e1")

    retriever = HypergraphRetriever()
    res = retriever.find_causal_chain(hg, premises={"hyp"}, target="access")

    assert res.found is False
    assert "secret_key" in res.explanatory_gap


def test_prompt_formatter():
    hg = Hypergraph()
    hg.add_hyperedge("Rel", tail={"A", "B"}, head={"C"}, edge_id="e1", relation_type=RelationType.CAUSAL)
    retriever = HypergraphRetriever()
    verts, edges = retriever.expand_subgraph(hg, ["A"])
    text = retriever.format_for_prompt(verts, edges)

    assert "Retrieved Hypergraph Memory Context" in text
    assert "`A`" in text
    assert "`C`" in text
    assert "CAUSAL" in text
