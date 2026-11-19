from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from agent_hypergraph.models import Hyperedge, Hypergraph, RelationType


@dataclass
class EpistemicConflict:
    conflict_id: str
    edge_a_id: str
    edge_b_id: str
    shared_vertices: List[str]
    tension_score: float
    description: str
    agents_involved: List[str] = field(default_factory=list)


class ConflictDetector:
    """Detects epistemic contradictions and hyperclique tension across multi-agent assertions."""

    def __init__(self, jaccard_threshold: float = 0.5) -> None:
        self.jaccard_threshold = jaccard_threshold

    def _jaccard_similarity(self, a: Set[str], b: Set[str]) -> float:
        union = a | b
        if not union:
            return 0.0
        return len(a & b) / len(union)

    def detect_conflicts(self, hg: Hypergraph) -> List[EpistemicConflict]:
        """Scans hyperedges for contradictory claims over shared entities."""
        conflicts: List[EpistemicConflict] = []
        edges = list(hg.hyperedges.values())
        n = len(edges)

        for i in range(n):
            for j in range(i + 1, n):
                e1 = edges[i]
                e2 = edges[j]

                # Check if edges share meaningful vertex overlap
                intersection = e1.members & e2.members
                if len(intersection) < 1:
                    continue

                jaccard = self._jaccard_similarity(e1.members, e2.members)
                if jaccard < self.jaccard_threshold:
                    continue

                # Check for explicit polarity contradiction in metadata
                pol1 = e1.metadata.get("polarity", None)
                pol2 = e2.metadata.get("polarity", None)
                val1 = e1.metadata.get("value", None)
                val2 = e2.metadata.get("value", None)

                is_conflict = False
                desc = ""

                if pol1 is not None and pol2 is not None and pol1 != pol2:
                    is_conflict = True
                    desc = f"Direct polarity contradiction between {e1.id} ({pol1}) and {e2.id} ({pol2})"

                elif val1 is not None and val2 is not None and val1 != val2:
                    is_conflict = True
                    desc = f"Incompatible assigned values: '{val1}' vs '{val2}'"

                elif (
                    e1.relation_type == RelationType.ASSERTION
                    and e2.relation_type == RelationType.ASSERTION
                    and e1.source_agent
                    and e2.source_agent
                    and e1.source_agent != e2.source_agent
                    and e1.metadata.get("negates", False) != e2.metadata.get("negates", False)
                ):
                    is_conflict = True
                    desc = f"Opposing agent assertions between {e1.source_agent} and {e2.source_agent}"

                if is_conflict:
                    # Tension score is function of overlap and confidences
                    tension = (e1.confidence + e2.confidence) / 2.0 * jaccard
                    agents = [a for a in [e1.source_agent, e2.source_agent] if a]
                    conflicts.append(
                        EpistemicConflict(
                            conflict_id=f"conf_{e1.id}_{e2.id}",
                            edge_a_id=e1.id,
                            edge_b_id=e2.id,
                            shared_vertices=sorted(list(intersection)),
                            tension_score=tension,
                            description=desc,
                            agents_involved=agents,
                        )
                    )

        return conflicts

    def compute_hyperclique_tension(self, hg: Hypergraph) -> float:
        """Computes aggregate epistemic tension score for the entire hypergraph."""
        conflicts = self.detect_conflicts(hg)
        if not conflicts:
            return 0.0
        return sum(c.tension_score for c in conflicts)

    def resolve_conflict(
        self,
        hg: Hypergraph,
        conflict: EpistemicConflict,
        agent_reputations: Optional[Dict[str, float]] = None,
    ) -> Optional[str]:
        """Resolves conflict by comparing confidence and agent reputation, dampening the losing edge."""
        e1 = hg.get_hyperedge(conflict.edge_a_id)
        e2 = hg.get_hyperedge(conflict.edge_b_id)
        if not e1 or not e2:
            return None

        rep = agent_reputations or {}
        r1 = rep.get(e1.source_agent or "", 1.0)
        r2 = rep.get(e2.source_agent or "", 1.0)

        score1 = e1.confidence * e1.weight * r1
        score2 = e2.confidence * e2.weight * r2

        if score1 >= score2:
            winner = e1.id
            # Dampen loser
            e2.weight *= 0.2
            e2.confidence *= 0.5
        else:
            winner = e2.id
            # Dampen loser
            e1.weight *= 0.2
            e1.confidence *= 0.5

        return winner
