#!/usr/bin/env python3
"""
Example: Multi-agent swarm collaborative memory and higher-order causal reasoning.
"""

from agent_hypergraph import (
    Hypergraph,
    HypergraphRetriever,
    RelationType,
    TemporalConsolidator,
    VertexType,
    compute_hypergraph_pagerank,
)


def main():
    hg = Hypergraph("swarm_triage_memory")

    # 1. Ingest entities, tools, and agents
    hg.add_vertex("Nova Coordinator", vertex_type=VertexType.AGENT, vertex_id="nova")
    hg.add_vertex("Sentinel Security", vertex_type=VertexType.AGENT, vertex_id="sentinel")
    hg.add_vertex("Grep Tool", vertex_type=VertexType.TOOL, vertex_id="tool_grep")
    hg.add_vertex("Auth Token Leak", vertex_type=VertexType.CLAIM, vertex_id="auth_leak")
    hg.add_vertex("Production Gateway", vertex_type=VertexType.ENTITY, vertex_id="prod_gateway")
    hg.add_vertex("Revoke API Key Action", vertex_type=VertexType.CONCEPT, vertex_id="action_revoke")

    # 2. Ingest multi-way relational hyperedges (connecting >2 entities simultaneously)
    # Edge 1: Sentinel used Grep Tool on Prod Gateway -> discovered Auth Token Leak
    hg.add_hyperedge(
        label="Vulnerability Discovery",
        tail={"sentinel", "tool_grep", "prod_gateway"},
        head={"auth_leak"},
        relation_type=RelationType.CAUSAL,
        confidence=0.98,
        source_agent="sentinel",
    )

    # Edge 2: Nova Coordinator reviews Auth Token Leak -> decides to Revoke API Key
    hg.add_hyperedge(
        label="Remediation Decision",
        tail={"nova", "auth_leak"},
        head={"action_revoke"},
        relation_type=RelationType.CAUSAL,
        confidence=0.95,
        source_agent="nova",
    )

    print("=== Hypergraph Memory Ingested ===")
    print(f"Total Vertices: {len(hg.vertices)}")
    print(f"Total Hyperedges: {len(hg.hyperedges)}")

    # 3. Compute Spectral Centrality (Hypergraph PageRank)
    ranks = compute_hypergraph_pagerank(hg)
    print("\n=== Top 3 Knowledge Hubs (PageRank) ===")
    for vid, score in list(ranks.items())[:3]:
        v = hg.get_vertex(vid)
        lbl = v.label if v else vid
        print(f"  • {lbl} ({vid}): {score:.4f}")

    # 4. Multi-hop Causal Chain Query
    retriever = HypergraphRetriever()
    res = retriever.find_causal_chain(hg, premises={"sentinel", "tool_grep", "prod_gateway", "nova"}, target="action_revoke")
    print("\n=== Causal Path Verification ===")
    print(f"Causal path found: {res.found}")
    print(f"Chain steps: {len(res.path_edges)}")
    for e in res.path_edges:
        print(f"  -> Hyperedge: {e.label} (tail: {sorted(e.tail)} -> head: {sorted(e.head)})")

    # 5. Temporal Decay and Ebbinghaus Consolidation
    consolidator = TemporalConsolidator(half_life_seconds=3600.0)
    report = consolidator.consolidate(hg)
    print("\n=== Memory Consolidation ===")
    print(f"Active memory nodes: {report['remaining_vertices']}, edges: {report['remaining_edges']}")


if __name__ == "__main__":
    main()
