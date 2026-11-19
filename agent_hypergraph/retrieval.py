from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from agent_hypergraph.models import Hyperedge, Hypergraph, Vertex


@dataclass
class CausalPathResult:
    found: bool
    path_edges: List[Hyperedge]
    visited_vertices: List[str]
    explanatory_gap: List[str]


class HypergraphRetriever:
    """Retrieves relevant semantic subgraphs and causal reasoning paths for LLMs."""

    def __init__(self, default_max_hops: int = 2, default_max_nodes: int = 50) -> None:
        self.default_max_hops = default_max_hops
        self.default_max_nodes = default_max_nodes

    def expand_subgraph(
        self,
        hg: Hypergraph,
        seed_vertices: List[str],
        max_hops: Optional[int] = None,
        max_nodes: Optional[int] = None,
        min_weight: float = 0.05,
    ) -> Tuple[Dict[str, Vertex], Dict[str, Hyperedge]]:
        """Expands around seed vertices via incident hyperedges up to max_hops."""
        hops_limit = max_hops if max_hops is not None else self.default_max_hops
        nodes_limit = max_nodes if max_nodes is not None else self.default_max_nodes

        found_vertices: Dict[str, Vertex] = {}
        found_edges: Dict[str, Hyperedge] = {}

        queue: deque[Tuple[str, int]] = deque()
        visited: Set[str] = set()

        for s in seed_vertices:
            v = hg.get_vertex(s)
            if v:
                found_vertices[s] = v
                queue.append((s, 0))
                visited.add(s)

        while queue and len(found_vertices) < nodes_limit:
            curr_vid, depth = queue.popleft()
            if depth >= hops_limit:
                continue

            incident = hg.incident_hyperedges(curr_vid)
            for edge in incident:
                if edge.weight < min_weight:
                    continue
                found_edges[edge.id] = edge

                for nbr_id in edge.members:
                    if nbr_id not in visited and len(found_vertices) < nodes_limit:
                        nbr_v = hg.get_vertex(nbr_id)
                        if nbr_v:
                            found_vertices[nbr_id] = nbr_v
                            visited.add(nbr_id)
                            queue.append((nbr_id, depth + 1))

        # Touch retrieved edges to record access
        hg.touch(list(found_edges.keys()))

        return found_vertices, found_edges

    def find_causal_chain(
        self,
        hg: Hypergraph,
        premises: Set[str],
        target: str,
        max_depth: int = 6,
    ) -> CausalPathResult:
        """Finds a sequence of directed hyperedges leading from premise concepts to the target."""
        if target in premises:
            return CausalPathResult(found=True, path_edges=[], visited_vertices=[target], explanatory_gap=[])

        known_state = set(premises)
        active_edges: List[Hyperedge] = []
        visited = list(premises)

        for _ in range(max_depth):
            progress = False
            # Find any directed hyperedge whose tail is completely satisfied by known_state
            for edge in hg.hyperedges.values():
                if edge.is_directed and edge not in active_edges:
                    if edge.tail.issubset(known_state):
                        active_edges.append(edge)
                        known_state.update(edge.head)
                        for h in edge.head:
                            if h not in visited:
                                visited.append(h)
                        progress = True
                        if target in known_state:
                            return CausalPathResult(
                                found=True,
                                path_edges=active_edges,
                                visited_vertices=visited,
                                explanatory_gap=[],
                            )
            if not progress:
                break

        # If not fully reached, identify missing dependencies
        missing = set()
        for edge in hg.hyperedges.values():
            if target in edge.head:
                missing.update(edge.tail - known_state)

        return CausalPathResult(
            found=False,
            path_edges=active_edges,
            visited_vertices=visited,
            explanatory_gap=sorted(list(missing)),
        )

    def format_for_prompt(
        self,
        vertices: Dict[str, Vertex],
        hyperedges: Dict[str, Hyperedge],
    ) -> str:
        """Formats the retrieved hypergraph into a concise Markdown section for LLM context."""
        lines = ["### Retrieved Hypergraph Memory Context", ""]
        lines.append("#### Participating Entities & Concepts:")
        for v in vertices.values():
            lines.append(f"- **[{v.vertex_type.value.upper()}]** `{v.id}` ({v.label})")

        lines.append("")
        lines.append("#### Multi-Way Relational Hyperedges:")
        for e in hyperedges.values():
            rel = e.relation_type.value.upper()
            if e.is_directed:
                tail_str = ", ".join(f"`{t}`" for t in sorted(e.tail))
                head_str = ", ".join(f"`{h}`" for h in sorted(e.head))
                lines.append(f"- **({rel})** [{tail_str}] ➔ [{head_str}] (confidence: {e.confidence:.2f}, weight: {e.weight:.2f})")
            else:
                mems = ", ".join(f"`{m}`" for m in sorted(e.members))
                lines.append(f"- **({rel})** {{{mems}}} (confidence: {e.confidence:.2f}, weight: {e.weight:.2f})")

        return "\n".join(lines)
