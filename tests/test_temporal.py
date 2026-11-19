import time
import pytest
from agent_hypergraph.models import Hypergraph, RelationType
from agent_hypergraph.temporal import DecayModel, TemporalConsolidator


def test_decay_models_calculation():
    tc_exp = TemporalConsolidator(decay_model=DecayModel.EXPONENTIAL, half_life_seconds=100.0)
    tc_pow = TemporalConsolidator(decay_model=DecayModel.POWER_LAW, half_life_seconds=100.0)
    tc_lin = TemporalConsolidator(decay_model=DecayModel.LINEAR, half_life_seconds=100.0)

    # After 100 seconds (1 half life), exponential should halve
    w_exp = tc_exp.calculate_decay(original_weight=1.0, decay_rate=0.05, elapsed_seconds=100.0)
    assert w_exp == pytest.approx(0.5, rel=1e-2)

    w_pow = tc_pow.calculate_decay(original_weight=1.0, decay_rate=0.05, elapsed_seconds=100.0)
    assert w_pow < 1.0

    w_lin = tc_lin.calculate_decay(original_weight=1.0, decay_rate=0.05, elapsed_seconds=100.0)
    assert w_lin == pytest.approx(0.5, rel=1e-2)


def test_consolidation_and_pruning():
    hg = Hypergraph()
    t0 = 1000.0
    # Fresh edge
    e1 = hg.add_hyperedge("e1", tail={"a", "b"}, edge_id="e1", weight=1.0)
    e1.last_accessed_at = t0 + 950.0  # accessed 50s ago

    # Stale edge with fast decay
    e2 = hg.add_hyperedge("e2", tail={"c", "d"}, edge_id="e2", weight=0.1, decay_rate=0.5)
    e2.last_accessed_at = t0  # accessed 1000s ago

    tc = TemporalConsolidator(half_life_seconds=100.0, prune_threshold=0.05)
    report = tc.consolidate(hg, current_time=t0 + 1000.0)

    assert report["pruned_edge_count"] == 1
    assert "e2" in report["pruned_edge_ids"]
    assert "e1" in hg.hyperedges
    assert "e2" not in hg.hyperedges


def test_reinforcement():
    hg = Hypergraph()
    e = hg.add_hyperedge("task", tail={"dev", "code"}, edge_id="e1", weight=1.0)
    tc = TemporalConsolidator(access_boost=0.5, max_weight=3.0)

    count = tc.reinforce(hg, ["e1", "nonexistent"])
    assert count == 1
    assert e.weight == pytest.approx(1.5)


def test_promote_episodic_to_semantic():
    hg = Hypergraph()
    # 3 episodic encounters involving {agent_x, tool_y, db_z}
    hg.add_hyperedge("Ep1", tail={"agent_x", "tool_y", "db_z"}, relation_type=RelationType.TEMPORAL, edge_id="ep1")
    hg.add_hyperedge("Ep2", tail={"agent_x", "tool_y", "db_z"}, relation_type=RelationType.TEMPORAL, edge_id="ep2")
    hg.add_hyperedge("Ep3", tail={"agent_x", "tool_y", "db_z"}, relation_type=RelationType.TEMPORAL, edge_id="ep3")

    tc = TemporalConsolidator()
    promoted = tc.promote_episodic_to_semantic(hg, min_support=2)

    assert len(promoted) == 1
    sem = promoted[0]
    assert sem.relation_type == RelationType.DERIVATION
    assert sem.members == {"agent_x", "tool_y", "db_z"}
    assert sem.decay_rate == 0.01  # Slower decay
