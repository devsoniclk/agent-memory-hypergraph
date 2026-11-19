import pytest
from agent_hypergraph.conflict import ConflictDetector, EpistemicConflict
from agent_hypergraph.models import Hypergraph, RelationType


def test_conflict_detection_polarity():
    hg = Hypergraph()
    # Agent 1 asserts service is healthy
    hg.add_hyperedge(
        "Claim healthy",
        tail={"auth_service", "db"},
        relation_type=RelationType.ASSERTION,
        source_agent="AgentAlpha",
        metadata={"polarity": True},
        edge_id="e_pos",
        confidence=0.9,
    )
    # Agent 2 asserts service is unhealthy
    hg.add_hyperedge(
        "Claim down",
        tail={"auth_service", "db"},
        relation_type=RelationType.ASSERTION,
        source_agent="AgentBeta",
        metadata={"polarity": False},
        edge_id="e_neg",
        confidence=0.8,
    )

    detector = ConflictDetector(jaccard_threshold=0.5)
    conflicts = detector.detect_conflicts(hg)

    assert len(conflicts) == 1
    c = conflicts[0]
    assert c.edge_a_id == "e_pos"
    assert c.edge_b_id == "e_neg"
    assert "auth_service" in c.shared_vertices
    assert c.tension_score > 0.0

    tension = detector.compute_hyperclique_tension(hg)
    assert tension == pytest.approx(c.tension_score)


def test_conflict_resolution():
    hg = Hypergraph()
    hg.add_hyperedge(
        "Positive claim",
        tail={"target"},
        metadata={"polarity": True},
        source_agent="TrustedAgent",
        edge_id="e1",
        confidence=0.95,
        weight=1.0,
    )
    hg.add_hyperedge(
        "Negative claim",
        tail={"target"},
        metadata={"polarity": False},
        source_agent="RogueAgent",
        edge_id="e2",
        confidence=0.6,
        weight=1.0,
    )

    detector = ConflictDetector(jaccard_threshold=0.5)
    conflicts = detector.detect_conflicts(hg)
    assert len(conflicts) == 1

    winner = detector.resolve_conflict(hg, conflicts[0], agent_reputations={"TrustedAgent": 1.5, "RogueAgent": 0.5})
    assert winner == "e1"

    # Loser e2 should have dampened weight and confidence
    e2 = hg.get_hyperedge("e2")
    assert e2.weight < 0.5
    assert e2.confidence < 0.5
