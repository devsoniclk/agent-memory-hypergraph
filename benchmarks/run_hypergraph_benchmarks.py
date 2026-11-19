#!/usr/bin/env python3
"""
Performance benchmarks for agent-memory-hypergraph:
1. Ingestion throughput (vertices and n-ary hyperedges per second)
2. Hypergraph PageRank power iteration latency
3. Multi-hop neighborhood retrieval latency
"""

import time
from agent_hypergraph import (
    Hypergraph,
    HypergraphRetriever,
    RelationType,
    compute_hypergraph_pagerank,
)


def benchmark_ingestion_and_spectral():
    hg = Hypergraph("benchmark_graph")
    num_nodes = 1000
    num_edges = 500

    print(f"Benchmarking ingestion of {num_nodes} vertices and {num_edges} 4-ary hyperedges...")
    t0 = time.perf_counter()
    for i in range(num_nodes):
        hg.add_vertex(f"Concept_{i}", vertex_id=f"v_{i}")

    for j in range(num_edges):
        tail = {f"v_{(j * 3 + k) % num_nodes}" for k in range(3)}
        head = {f"v_{(j * 3 + 3) % num_nodes}"}
        hg.add_hyperedge(f"Edge_{j}", tail=tail, head=head, relation_type=RelationType.CAUSAL)

    t1 = time.perf_counter()
    ingest_time = t1 - t0
    print(f"  Ingestion time: {ingest_time*1000:.2f} ms ({num_edges / ingest_time:.0f} edges/sec)")

    print("Benchmarking Hypergraph PageRank power iteration...")
    t2 = time.perf_counter()
    ranks = compute_hypergraph_pagerank(hg, damping=0.85, max_iter=50)
    t3 = time.perf_counter()
    pr_time = t3 - t2
    print(f"  PageRank latency: {pr_time*1000:.2f} ms for {len(ranks)} nodes")

    print("Benchmarking 2-hop neighborhood retrieval across 50 queries...")
    retriever = HypergraphRetriever(default_max_hops=2)
    t4 = time.perf_counter()
    for q in range(50):
        seed = f"v_{(q * 17) % num_nodes}"
        retriever.expand_subgraph(hg, [seed], max_hops=2)
    t5 = time.perf_counter()
    query_time = (t5 - t4) / 50.0
    print(f"  Average 2-hop query latency: {query_time*1000:.2f} ms / query")


if __name__ == "__main__":
    benchmark_ingestion_and_spectral()
