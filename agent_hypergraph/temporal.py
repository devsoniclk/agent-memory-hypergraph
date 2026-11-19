from __future__ import annotations

import math
import time
from collections import defaultdict
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

from agent_hypergraph.models import Hyperedge, Hypergraph, RelationType, VertexType


class DecayModel(str, Enum):
    EXPONENTIAL = "exponential"
    POWER_LAW = "power_law"
    LINEAR = "linear"


class TemporalConsolidator:
    """Manages memory decay, reinforcement, episodic consolidation, and compaction."""

    def __init__(
        self,
        decay_model: DecayModel = DecayModel.EXPONENTIAL,
        half_life_seconds: float = 3600.0,  # 1 hour default
        prune_threshold: float = 0.05,
        access_boost: float = 0.25,
        max_weight: float = 10.0,
    ) -> None:
        self.decay_model = decay_model
        self.half_life_seconds = max(1.0, half_life_seconds)
        self.prune_threshold = prune_threshold
        self.access_boost = access_boost
        self.max_weight = max_weight

    def calculate_decay(self, original_weight: float, decay_rate: float, elapsed_seconds: float) -> float:
        """Calculates effective decayed weight after elapsed time."""
        if elapsed_seconds <= 0:
            return original_weight

        if self.decay_model == DecayModel.EXPONENTIAL:
            # lambda = ln(2) / half_life * (decay_rate / 0.05)
            rate = (math.log(2) / self.half_life_seconds) * (decay_rate / 0.05)
            return original_weight * math.exp(-rate * elapsed_seconds)

        elif self.decay_model == DecayModel.POWER_LAW:
            scale = 1.0 + (elapsed_seconds / self.half_life_seconds) * (decay_rate / 0.05)
            return original_weight / (scale ** 0.5)

        elif self.decay_model == DecayModel.LINEAR:
            decay_amount = (elapsed_seconds / self.half_life_seconds) * original_weight * 0.5
            return max(0.0, original_weight - decay_amount)

        return original_weight

    def consolidate(
        self,
        hg: Hypergraph,
        current_time: Optional[float] = None,
        protected_edge_ids: Optional[Set[str]] = None,
    ) -> Dict[str, Any]:
        """Applies temporal decay to all hyperedges and prunes sub-threshold edges."""
        now = current_time if current_time is not None else time.time()
        protected = protected_edge_ids or set()

        pruned_edges = []
        decayed_counts = 0

        for eid, edge in list(hg.hyperedges.items()):
            if eid in protected or edge.metadata.get("permanent", False):
                continue

            elapsed = max(0.0, now - edge.last_accessed_at)
            decayed = self.calculate_decay(edge.weight, edge.decay_rate, elapsed)
            edge.weight = max(0.0, decayed)
            decayed_counts += 1

            if edge.weight < self.prune_threshold:
                pruned_edges.append(eid)
                hg.remove_hyperedge(eid)

        # Check for isolated vertices with degree 0
        pruned_vertices = []
        for vid in list(hg.vertices.keys()):
            if hg.vertex_degree(vid) == 0.0 and vid not in protected:
                pruned_vertices.append(vid)
                hg.remove_vertex(vid)

        return {
            "decayed_edge_count": decayed_counts,
            "pruned_edge_count": len(pruned_edges),
            "pruned_vertex_count": len(pruned_vertices),
            "pruned_edge_ids": pruned_edges,
            "pruned_vertex_ids": pruned_vertices,
            "remaining_edges": len(hg.hyperedges),
            "remaining_vertices": len(hg.vertices),
        }

    def reinforce(
        self,
        hg: Hypergraph,
        edge_ids: List[str],
        boost: Optional[float] = None,
    ) -> int:
        """Reinforces accessed hyperedges by boosting weight and updating access timestamp."""
        b = boost if boost is not None else self.access_boost
        now = time.time()
        reinforced = 0

        for eid in edge_ids:
            edge = hg.get_hyperedge(eid)
            if edge:
                edge.weight = min(self.max_weight, edge.weight + b)
                edge.last_accessed_at = now
                reinforced += 1

        return reinforced

    def promote_episodic_to_semantic(
        self,
        hg: Hypergraph,
        min_support: int = 2,
    ) -> List[Hyperedge]:
        """Finds recurring co-activated episodic subsets and synthesizes consolidated semantic hyperedges."""
        # Group episodes by common vertex subsets of size >= 2
        subset_counts: Dict[Tuple[str, ...], List[Hyperedge]] = defaultdict(list)

        for edge in hg.hyperedges.values():
            if edge.relation_type == RelationType.TEMPORAL or edge.relation_type == RelationType.ASSERTION:
                members = sorted(list(edge.members))
                if len(members) >= 2:
                    subset_counts[tuple(members)].append(edge)

        promoted = []
        for subset, edges in subset_counts.items():
            if len(edges) >= min_support:
                semantic_id = f"sem_{abs(hash(subset)) % 100000000}"
                if semantic_id not in hg.hyperedges:
                    avg_conf = sum(e.confidence for e in edges) / len(edges)
                    total_weight = min(self.max_weight, sum(e.weight for e in edges) * 0.75)
                    label = f"Consolidated concept: {', '.join(subset)}"
                    sem_edge = hg.add_hyperedge(
                        label=label,
                        tail=set(subset),
                        relation_type=RelationType.DERIVATION,
                        edge_id=semantic_id,
                        weight=total_weight,
                        confidence=avg_conf,
                        decay_rate=0.01,  # Semantic memory decays much slower
                        metadata={"consolidated_from": [e.id for e in edges]},
                    )
                    promoted.append(sem_edge)

        return promoted
