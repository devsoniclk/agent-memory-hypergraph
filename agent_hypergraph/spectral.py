from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Set, Tuple

from agent_hypergraph.models import Hypergraph


def hypergraph_transition_probability(hg: Hypergraph, u: str, v: str) -> float:
    """Computes two-step random walk probability from vertex u to vertex v via incident hyperedges."""
    if u == v or u not in hg.vertices or v not in hg.vertices:
        return 0.0

    incident_edges = hg.incident_hyperedges(u)
    total_u_weight = sum(e.weight * e.confidence for e in incident_edges)
    if total_u_weight <= 0.0:
        return 0.0

    prob = 0.0
    for edge in incident_edges:
        if v in edge.members:
            edge_weight = edge.weight * edge.confidence
            prob_edge = edge_weight / total_u_weight
            # Cardinality minus current vertex
            cardinality = len(edge.members) - 1
            if cardinality > 0:
                prob += prob_edge * (1.0 / cardinality)

    return prob


def hypergraph_random_walk(
    hg: Hypergraph,
    start_vertex: str,
    walk_length: int = 20,
    restart_prob: float = 0.15,
    seed: Optional[int] = None,
) -> List[str]:
    """Generates a random walk trajectory across hyperedges with random restart."""
    if seed is not None:
        random.seed(seed)

    if start_vertex not in hg.vertices:
        return []

    path = [start_vertex]
    curr = start_vertex

    for _ in range(walk_length - 1):
        if random.random() < restart_prob:
            curr = start_vertex
            path.append(curr)
            continue

        edges = hg.incident_hyperedges(curr)
        if not edges:
            curr = start_vertex
            path.append(curr)
            continue

        weights = [max(1e-6, e.weight * e.confidence) for e in edges]
        selected_edge = random.choices(edges, weights=weights, k=1)[0]

        # In directed edge, try to move from tail to head
        candidates = list(selected_edge.members - {curr})
        if selected_edge.is_directed and curr in selected_edge.tail and selected_edge.head:
            head_candidates = list(selected_edge.head - {curr})
            if head_candidates:
                candidates = head_candidates

        if not candidates:
            curr = start_vertex
        else:
            curr = random.choice(candidates)
        path.append(curr)

    return path


def compute_hypergraph_pagerank(
    hg: Hypergraph,
    damping: float = 0.85,
    max_iter: int = 100,
    tol: float = 1e-6,
    personalization: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """Computes higher-order Hypergraph PageRank using power iteration."""
    vertices = list(hg.vertices.keys())
    n = len(vertices)
    if n == 0:
        return {}
    if n == 1:
        return {vertices[0]: 1.0}

    # Teleport vector (uniform or personalized)
    p_vec: Dict[str, float] = {}
    if personalization:
        p_sum = sum(personalization.get(v, 0.0) for v in vertices)
        if p_sum > 0:
            p_vec = {v: personalization.get(v, 0.0) / p_sum for v in vertices}
    if not p_vec:
        p_vec = {v: 1.0 / n for v in vertices}

    # Initialize ranks
    rank = {v: 1.0 / n for v in vertices}

    # Precompute transition matrix out-degrees and adjacencies
    out_edges: Dict[str, List[Tuple[str, float]]] = {}
    for u in vertices:
        trans_list = []
        incident = hg.incident_hyperedges(u)
        sum_w = sum(e.weight * e.confidence for e in incident)
        if sum_w > 0:
            # Map neighbor v -> cumulative weight
            nbr_weights: Dict[str, float] = {}
            for e in incident:
                w_e = (e.weight * e.confidence) / sum_w
                denom = len(e.members) - 1
                if denom > 0:
                    for v in e.members:
                        if v != u:
                            nbr_weights[v] = nbr_weights.get(v, 0.0) + w_e / denom
            for v, prob in nbr_weights.items():
                trans_list.append((v, prob))
        out_edges[u] = trans_list

    for _ in range(max_iter):
        next_rank = {v: (1.0 - damping) * p_vec[v] for v in vertices}
        dangling_sum = 0.0

        for u in vertices:
            trans = out_edges[u]
            if not trans:
                dangling_sum += rank[u]
            else:
                for v, prob in trans:
                    next_rank[v] += damping * rank[u] * prob

        # Re-distribute dangling node mass
        if dangling_sum > 0:
            for v in vertices:
                next_rank[v] += damping * dangling_sum * p_vec[v]

        diff = sum(abs(next_rank[v] - rank[v]) for v in vertices)
        rank = next_rank
        if diff < tol:
            break

    # Normalize output
    total = sum(rank.values())
    if total > 0:
        return {v: score / total for v, score in sorted(rank.items(), key=lambda x: x[1], reverse=True)}
    return rank


def compute_hypergraph_laplacian(hg: Hypergraph) -> Dict[str, Any]:
    """Generates the incidence matrix and normalized hypergraph Laplacian spectral properties."""
    v_ids = sorted(list(hg.vertices.keys()))
    e_ids = sorted(list(hg.hyperedges.keys()))
    n_v = len(v_ids)
    n_e = len(e_ids)

    v_map = {vid: i for i, vid in enumerate(v_ids)}
    e_map = {eid: j for j, eid in enumerate(e_ids)}

    # Degrees
    d_v = [0.0] * n_v
    d_e = [0.0] * n_e
    weights_e = [hg.hyperedges[eid].weight * hg.hyperedges[eid].confidence for eid in e_ids]

    incidence: List[List[float]] = [[0.0] * n_e for _ in range(n_v)]
    for eid, edge in hg.hyperedges.items():
        j = e_map[eid]
        w = edge.weight * edge.confidence
        d_e[j] = float(len(edge.members))
        for vid in edge.members:
            if vid in v_map:
                i = v_map[vid]
                incidence[i][j] = 1.0
                d_v[i] += w

    return {
        "num_vertices": n_v,
        "num_hyperedges": n_e,
        "vertices": v_ids,
        "hyperedges": e_ids,
        "vertex_degrees": {v_ids[i]: d_v[i] for i in range(n_v)},
        "hyperedge_ranks": {e_ids[j]: int(d_e[j]) for j in range(n_e)},
    }
