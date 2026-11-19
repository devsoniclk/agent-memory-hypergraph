import pytest
from agent_hypergraph.models import Hypergraph, Hyperedge, RelationType, Vertex, VertexType


def test_vertex_creation_and_dict():
    v = Vertex(id="v1", label="Auth Service", vertex_type=VertexType.ENTITY, properties={"lang": "go"})
    d = v.to_dict()
    assert d["id"] == "v1"
    assert d["vertex_type"] == "entity"
    assert d["properties"]["lang"] == "go"

    v_restored = Vertex.from_dict(d)
    assert v_restored.id == v.id
    assert v_restored.label == v.label
    assert v_restored.vertex_type == VertexType.ENTITY


def test_hyperedge_creation_and_properties():
    e = Hyperedge(
        id="e1",
        label="Causal tool interaction",
        tail={"agent1", "tool_grep"},
        head={"auth_vuln"},
        relation_type=RelationType.CAUSAL,
        confidence=0.95,
    )
    assert e.is_directed is True
    assert e.members == {"agent1", "tool_grep", "auth_vuln"}

    d = e.to_dict()
    assert d["relation_type"] == "causal"
    assert "agent1" in d["tail"]

    e_restored = Hyperedge.from_dict(d)
    assert e_restored.id == e.id
    assert e_restored.members == e.members


def test_hypergraph_add_and_lookup():
    hg = Hypergraph(name="test_graph")
    v1 = hg.add_vertex(label="Alice", vertex_type=VertexType.AGENT, vertex_id="alice")
    v2 = hg.add_vertex(label="Bob", vertex_type=VertexType.AGENT, vertex_id="bob")
    v3 = hg.add_vertex(label="Repo", vertex_type=VertexType.ENTITY, vertex_id="repo")

    edge = hg.add_hyperedge(
        label="Collaborative PR",
        tail={"alice", "bob"},
        head={"repo"},
        relation_type=RelationType.COLLABORATIVE,
        edge_id="e_collab",
    )

    assert hg.get_vertex("alice") == v1
    assert hg.get_hyperedge("e_collab") == edge
    assert hg.vertex_degree("alice") == 1.0
    assert hg.hyperedge_degree("e_collab") == 3
    assert "bob" in hg.neighbors("alice")
    assert "repo" in hg.neighbors("alice")


def test_hypergraph_auto_create_missing_vertices():
    hg = Hypergraph()
    edge = hg.add_hyperedge(
        label="Quick relation",
        tail={"nodeA", "nodeB"},
        head={"nodeC"},
    )
    assert "nodeA" in hg.vertices
    assert "nodeB" in hg.vertices
    assert "nodeC" in hg.vertices
    assert edge.members == {"nodeA", "nodeB", "nodeC"}


def test_hypergraph_removal():
    hg = Hypergraph()
    hg.add_hyperedge("rel", tail={"x", "y"}, head={"z"}, edge_id="e1")
    assert len(hg.hyperedges) == 1
    assert len(hg.neighbors("x")) == 2

    # Remove hyperedge
    assert hg.remove_hyperedge("e1") is True
    assert len(hg.hyperedges) == 0
    assert len(hg.neighbors("x")) == 0
    assert hg.remove_hyperedge("e1") is False

    # Remove vertex
    hg.add_hyperedge("rel2", tail={"a", "b"}, head={"c"}, edge_id="e2")
    assert hg.remove_vertex("a") is True
    assert "a" not in hg.vertices
    assert "a" not in hg.get_hyperedge("e2").members
    assert hg.remove_vertex("nonexistent") is False


def test_hypergraph_dict_roundtrip():
    hg = Hypergraph("memory")
    hg.add_vertex("Agent Nova", vertex_type=VertexType.AGENT, vertex_id="nova")
    hg.add_hyperedge("Audit", tail={"nova"}, head={"target_sys"}, edge_id="audit1")

    data = hg.to_dict()
    hg2 = Hypergraph.from_dict(data)
    assert hg2.name == "memory"
    assert "nova" in hg2.vertices
    assert "target_sys" in hg2.vertices
    assert "audit1" in hg2.hyperedges
    assert hg2.get_hyperedge("audit1").members == {"nova", "target_sys"}
