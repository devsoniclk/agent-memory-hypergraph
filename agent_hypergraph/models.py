from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple


class VertexType(str, Enum):
    ENTITY = "entity"
    CONCEPT = "concept"
    AGENT = "agent"
    TOOL = "tool"
    EPISODE = "episode"
    CLAIM = "claim"


class RelationType(str, Enum):
    CAUSAL = "causal"
    COLLABORATIVE = "collaborative"
    DERIVATION = "derivation"
    TEMPORAL = "temporal"
    ASSERTION = "assertion"
    DEPENDENCY = "dependency"


@dataclass
class Vertex:
    id: str
    label: str
    vertex_type: VertexType = VertexType.CONCEPT
    weight: float = 1.0
    properties: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "vertex_type": self.vertex_type.value,
            "weight": self.weight,
            "properties": self.properties,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Vertex:
        return cls(
            id=data["id"],
            label=data["label"],
            vertex_type=VertexType(data.get("vertex_type", "concept")),
            weight=float(data.get("weight", 1.0)),
            properties=data.get("properties", {}),
            created_at=float(data.get("created_at", time.time())),
            updated_at=float(data.get("updated_at", time.time())),
        )


@dataclass
class Hyperedge:
    id: str
    label: str
    tail: Set[str] = field(default_factory=set)  # source/premise vertices
    head: Set[str] = field(default_factory=set)  # target/consequence vertices
    relation_type: RelationType = RelationType.ASSERTION
    weight: float = 1.0
    confidence: float = 1.0
    decay_rate: float = 0.05
    last_accessed_at: float = field(default_factory=time.time)
    source_agent: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    @property
    def members(self) -> Set[str]:
        """Set of all participating vertices in this hyperedge."""
        return self.tail | self.head

    @property
    def is_directed(self) -> bool:
        return len(self.tail) > 0 and len(self.head) > 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "tail": sorted(list(self.tail)),
            "head": sorted(list(self.head)),
            "relation_type": self.relation_type.value,
            "weight": self.weight,
            "confidence": self.confidence,
            "decay_rate": self.decay_rate,
            "last_accessed_at": self.last_accessed_at,
            "source_agent": self.source_agent,
            "metadata": self.metadata,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Hyperedge:
        return cls(
            id=data["id"],
            label=data["label"],
            tail=set(data.get("tail", [])),
            head=set(data.get("head", [])),
            relation_type=RelationType(data.get("relation_type", "assertion")),
            weight=float(data.get("weight", 1.0)),
            confidence=float(data.get("confidence", 1.0)),
            decay_rate=float(data.get("decay_rate", 0.05)),
            last_accessed_at=float(data.get("last_accessed_at", time.time())),
            source_agent=data.get("source_agent"),
            metadata=data.get("metadata", {}),
            created_at=float(data.get("created_at", time.time())),
        )


class Hypergraph:
    """In-memory higher-order hypergraph structure with dual indexing."""

    def __init__(self, name: str = "default_hypergraph") -> None:
        self.name = name
        self.vertices: Dict[str, Vertex] = {}
        self.hyperedges: Dict[str, Hyperedge] = {}
        # Dual incidence tracking: vertex_id -> set of hyperedge_ids
        self._v_to_e: Dict[str, Set[str]] = {}

    def add_vertex(
        self,
        label: str,
        vertex_type: VertexType = VertexType.CONCEPT,
        vertex_id: Optional[str] = None,
        weight: float = 1.0,
        properties: Optional[Dict[str, Any]] = None,
    ) -> Vertex:
        v_id = vertex_id or f"v_{uuid.uuid4().hex[:12]}"
        if v_id in self.vertices:
            v = self.vertices[v_id]
            v.label = label
            v.vertex_type = vertex_type
            v.weight = weight
            if properties:
                v.properties.update(properties)
            v.updated_at = time.time()
            return v

        v = Vertex(
            id=v_id,
            label=label,
            vertex_type=vertex_type,
            weight=weight,
            properties=properties or {},
        )
        self.vertices[v_id] = v
        self._v_to_e[v_id] = set()
        return v

    def add_hyperedge(
        self,
        label: str,
        tail: Set[str],
        head: Optional[Set[str]] = None,
        relation_type: RelationType = RelationType.ASSERTION,
        edge_id: Optional[str] = None,
        weight: float = 1.0,
        confidence: float = 1.0,
        decay_rate: float = 0.05,
        source_agent: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Hyperedge:
        e_id = edge_id or f"e_{uuid.uuid4().hex[:12]}"
        h_set = head or set()
        all_participants = tail | h_set

        # Auto-create missing vertices if not present
        for vid in all_participants:
            if vid not in self.vertices:
                self.add_vertex(label=vid, vertex_id=vid)

        edge = Hyperedge(
            id=e_id,
            label=label,
            tail=set(tail),
            head=set(h_set),
            relation_type=relation_type,
            weight=weight,
            confidence=max(0.0, min(1.0, confidence)),
            decay_rate=decay_rate,
            source_agent=source_agent,
            metadata=metadata or {},
        )
        self.hyperedges[e_id] = edge

        for vid in all_participants:
            self._v_to_e.setdefault(vid, set()).add(e_id)

        return edge

    def get_vertex(self, vertex_id: str) -> Optional[Vertex]:
        return self.vertices.get(vertex_id)

    def get_hyperedge(self, edge_id: str) -> Optional[Hyperedge]:
        return self.hyperedges.get(edge_id)

    def remove_hyperedge(self, edge_id: str) -> bool:
        if edge_id not in self.hyperedges:
            return False
        edge = self.hyperedges.pop(edge_id)
        for vid in edge.members:
            if vid in self._v_to_e:
                self._v_to_e[vid].discard(edge_id)
        return True

    def remove_vertex(self, vertex_id: str) -> bool:
        if vertex_id not in self.vertices:
            return False
        self.vertices.pop(vertex_id)
        incident_edges = list(self._v_to_e.pop(vertex_id, set()))
        for eid in incident_edges:
            if eid in self.hyperedges:
                edge = self.hyperedges[eid]
                edge.tail.discard(vertex_id)
                edge.head.discard(vertex_id)
                if len(edge.members) == 0:
                    self.hyperedges.pop(eid, None)
        return True

    def vertex_degree(self, vertex_id: str) -> float:
        """Weighted degree: sum of weights of incident hyperedges."""
        if vertex_id not in self._v_to_e:
            return 0.0
        return sum(self.hyperedges[eid].weight for eid in self._v_to_e[vertex_id] if eid in self.hyperedges)

    def hyperedge_degree(self, edge_id: str) -> int:
        """Cardinality / rank of hyperedge: count of member vertices."""
        if edge_id not in self.hyperedges:
            return 0
        return len(self.hyperedges[edge_id].members)

    def incident_hyperedges(self, vertex_id: str) -> List[Hyperedge]:
        if vertex_id not in self._v_to_e:
            return []
        return [self.hyperedges[eid] for eid in self._v_to_e[vertex_id] if eid in self.hyperedges]

    def neighbors(self, vertex_id: str) -> Set[str]:
        """All vertices sharing at least one hyperedge with this vertex."""
        nbrs = set()
        for edge in self.incident_hyperedges(vertex_id):
            nbrs.update(edge.members)
        nbrs.discard(vertex_id)
        return nbrs

    def touch(self, edge_ids: List[str]) -> None:
        """Reinforce / record access to hyperedges."""
        now = time.time()
        for eid in edge_ids:
            if eid in self.hyperedges:
                self.hyperedges[eid].last_accessed_at = now

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "vertices": [v.to_dict() for v in self.vertices.values()],
            "hyperedges": [e.to_dict() for e in self.hyperedges.values()],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Hypergraph:
        hg = cls(name=data.get("name", "restored_hypergraph"))
        for vdata in data.get("vertices", []):
            v = Vertex.from_dict(vdata)
            hg.vertices[v.id] = v
            hg._v_to_e[v.id] = set()
        for edata in data.get("hyperedges", []):
            e = Hyperedge.from_dict(edata)
            hg.hyperedges[e.id] = e
            for vid in e.members:
                hg._v_to_e.setdefault(vid, set()).add(e.id)
        return hg
