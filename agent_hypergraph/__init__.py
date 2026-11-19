"""
Agent Hypergraph: Higher-order hypergraph memory for autonomous AI agent swarms.
"""

from agent_hypergraph.models import (
    Vertex,
    VertexType,
    Hyperedge,
    RelationType,
    Hypergraph,
)
from agent_hypergraph.spectral import (
    compute_hypergraph_pagerank,
    hypergraph_random_walk,
    compute_hypergraph_laplacian,
)
from agent_hypergraph.temporal import (
    TemporalConsolidator,
    DecayModel,
)
from agent_hypergraph.retrieval import (
    HypergraphRetriever,
    CausalPathResult,
)
from agent_hypergraph.conflict import (
    ConflictDetector,
    EpistemicConflict,
)

__version__ = "0.1.0"
__all__ = [
    "Vertex",
    "VertexType",
    "Hyperedge",
    "RelationType",
    "Hypergraph",
    "compute_hypergraph_pagerank",
    "hypergraph_random_walk",
    "compute_hypergraph_laplacian",
    "TemporalConsolidator",
    "DecayModel",
    "HypergraphRetriever",
    "CausalPathResult",
    "ConflictDetector",
    "EpistemicConflict",
]
