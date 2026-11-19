from __future__ import annotations

import json
import os
import tempfile
from typing import Optional

from agent_hypergraph.models import Hypergraph


def save_to_json(hg: Hypergraph, file_path: str) -> None:
    """Atomically serializes hypergraph to JSON on disk."""
    data = hg.to_dict()
    dir_name = os.path.dirname(os.path.abspath(file_path))
    os.makedirs(dir_name, exist_ok=True)

    with tempfile.NamedTemporaryFile("w", dir=dir_name, delete=False) as tf:
        json.dump(data, tf, indent=2)
        temp_name = tf.name

    os.replace(temp_name, file_path)


def load_from_json(file_path: str) -> Hypergraph:
    """Deserializes hypergraph from JSON file."""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return Hypergraph.from_dict(data)


def export_to_mermaid(hg: Hypergraph, max_edges: int = 50) -> str:
    """Generates Mermaid flowchart diagram depicting hyperedges as hub vertices."""
    lines = ["flowchart TD"]

    # Render vertex styling
    for vid, v in hg.vertices.items():
        vtype = v.vertex_type.value
        label = v.label.replace('"', "'")
        lines.append(f'    {vid}["[{vtype}] {label}"]')

    # Render hyperedges as hubs
    count = 0
    for eid, e in hg.hyperedges.items():
        if count >= max_edges:
            break
        count += 1
        edge_hub = f"hub_{eid}"
        rel = e.relation_type.value.upper()
        lines.append(f'    {edge_hub}{{{{"{rel}: {e.label}"}}}}:::hyperhub')

        if e.is_directed:
            for t in e.tail:
                lines.append(f"    {t} --> {edge_hub}")
            for h in e.head:
                lines.append(f"    {edge_hub} --> {h}")
        else:
            for m in e.members:
                lines.append(f"    {m} --- {edge_hub}")

    lines.append("    classDef hyperhub fill:#e1f5fe,stroke:#0288d1,stroke-width:2px,color:#01579b;")
    return "\n".join(lines)


def export_to_dot(hg: Hypergraph) -> str:
    """Generates Graphviz DOT representation."""
    lines = ["digraph Hypergraph {", "    rankdir=LR;", "    node [shape=box, style=rounded];"]

    for vid, v in hg.vertices.items():
        lines.append(f'    "{vid}" [label="{v.label}\\n({v.vertex_type.value})"];')

    for eid, e in hg.hyperedges.items():
        hub = f"hub_{eid}"
        lines.append(f'    "{hub}" [shape=diamond, label="{e.relation_type.value}", style=filled, fillcolor=lightblue];')
        if e.is_directed:
            for t in e.tail:
                lines.append(f'    "{t}" -> "{hub}";')
            for h in e.head:
                lines.append(f'    "{hub}" -> "{h}";')
        else:
            for m in e.members:
                lines.append(f'    "{m}" -> "{hub}" [dir=none];')

    lines.append("}")
    return "\n".join(lines)
