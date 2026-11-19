from __future__ import annotations

import json
import sys
from typing import Any, Dict, List, Optional

from agent_hypergraph.conflict import ConflictDetector
from agent_hypergraph.models import Hypergraph, RelationType, VertexType
from agent_hypergraph.retrieval import HypergraphRetriever
from agent_hypergraph.spectral import compute_hypergraph_pagerank
from agent_hypergraph.storage import export_to_mermaid
from agent_hypergraph.temporal import TemporalConsolidator


class AgentHypergraphMCPServer:
    """Stdio JSON-RPC 2.0 MCP server for autonomous agent memory hypergraph."""

    def __init__(self, hg: Optional[Hypergraph] = None) -> None:
        self.hg = hg or Hypergraph(name="mcp_shared_memory")
        self.retriever = HypergraphRetriever()
        self.consolidator = TemporalConsolidator()
        self.conflict_detector = ConflictDetector()

    def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        req_id = request.get("id")
        method = request.get("method")
        params = request.get("params", {})

        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "agent-memory-hypergraph", "version": "0.1.0"},
                },
            }

        elif method == "tools/list":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "tools": [
                        {
                            "name": "add_vertex",
                            "description": "Add an entity, concept, or agent vertex to the hypergraph memory.",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "label": {"type": "string"},
                                    "vertex_type": {"type": "string", "enum": ["entity", "concept", "agent", "tool", "episode", "claim"]},
                                    "vertex_id": {"type": "string"},
                                    "properties": {"type": "object"},
                                },
                                "required": ["label"],
                            },
                        },
                        {
                            "name": "add_hyperedge",
                            "description": "Add an n-ary multi-way relationship connecting vertices simultaneously.",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "label": {"type": "string"},
                                    "tail": {"type": "array", "items": {"type": "string"}},
                                    "head": {"type": "array", "items": {"type": "string"}},
                                    "relation_type": {"type": "string", "enum": ["causal", "collaborative", "derivation", "temporal", "assertion", "dependency"]},
                                    "confidence": {"type": "number"},
                                    "source_agent": {"type": "string"},
                                    "metadata": {"type": "object"},
                                },
                                "required": ["label", "tail"],
                            },
                        },
                        {
                            "name": "query_subgraph",
                            "description": "Retrieve multi-hop neighborhood and causal pathways around seed vertices.",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "seed_vertices": {"type": "array", "items": {"type": "string"}},
                                    "max_hops": {"type": "integer", "default": 2},
                                    "format_prompt": {"type": "boolean", "default": True},
                                },
                                "required": ["seed_vertices"],
                            },
                        },
                        {
                            "name": "compute_pagerank",
                            "description": "Compute higher-order spectral Hypergraph PageRank centrality scores.",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "damping": {"type": "number", "default": 0.85},
                                },
                            },
                        },
                        {
                            "name": "consolidate_memory",
                            "description": "Apply temporal Ebbinghaus decay, reinforce accessed paths, and prune stale edges.",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "reinforce_edge_ids": {"type": "array", "items": {"type": "string"}},
                                },
                            },
                        },
                        {
                            "name": "detect_conflicts",
                            "description": "Scan multi-agent memory for contradictory assertions and compute hyperclique tension.",
                            "inputSchema": {"type": "object", "properties": {}},
                        },
                        {
                            "name": "export_graph",
                            "description": "Export the current hypergraph state to Mermaid diagram or JSON.",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "format": {"type": "string", "enum": ["mermaid", "json"], "default": "mermaid"},
                                },
                            },
                        },
                    ]
                },
            }

        elif method == "tools/call":
            tool_name = params.get("name")
            args = params.get("arguments", {})
            try:
                res = self._execute_tool(tool_name, args)
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {"content": [{"type": "text", "text": json.dumps(res, indent=2)}]},
                }
            except Exception as ex:
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {"code": -32603, "message": str(ex)},
                }

        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32601, "message": f"Method {method} not found"},
        }

    def _execute_tool(self, name: str, args: Dict[str, Any]) -> Any:
        if name == "add_vertex":
            v = self.hg.add_vertex(
                label=args["label"],
                vertex_type=VertexType(args.get("vertex_type", "concept")),
                vertex_id=args.get("vertex_id"),
                properties=args.get("properties"),
            )
            return {"status": "ok", "vertex": v.to_dict()}

        elif name == "add_hyperedge":
            e = self.hg.add_hyperedge(
                label=args["label"],
                tail=set(args["tail"]),
                head=set(args.get("head", [])),
                relation_type=RelationType(args.get("relation_type", "assertion")),
                confidence=float(args.get("confidence", 1.0)),
                source_agent=args.get("source_agent"),
                metadata=args.get("metadata"),
            )
            return {"status": "ok", "hyperedge": e.to_dict()}

        elif name == "query_subgraph":
            seeds = args["seed_vertices"]
            hops = int(args.get("max_hops", 2))
            verts, edges = self.retriever.expand_subgraph(self.hg, seeds, max_hops=hops)
            if args.get("format_prompt", True):
                formatted = self.retriever.format_for_prompt(verts, edges)
                return {
                    "status": "ok",
                    "num_vertices": len(verts),
                    "num_hyperedges": len(edges),
                    "prompt_context": formatted,
                }
            return {
                "status": "ok",
                "vertices": [v.to_dict() for v in verts.values()],
                "hyperedges": [e.to_dict() for e in edges.values()],
            }

        elif name == "compute_pagerank":
            damp = float(args.get("damping", 0.85))
            ranks = compute_hypergraph_pagerank(self.hg, damping=damp)
            return {"status": "ok", "pagerank": ranks}

        elif name == "consolidate_memory":
            reinforce_ids = args.get("reinforce_edge_ids", [])
            if reinforce_ids:
                self.consolidator.reinforce(self.hg, reinforce_ids)
            result = self.consolidator.consolidate(self.hg)
            return {"status": "ok", "consolidation": result}

        elif name == "detect_conflicts":
            conflicts = self.conflict_detector.detect_conflicts(self.hg)
            tension = self.conflict_detector.compute_hyperclique_tension(self.hg)
            return {
                "status": "ok",
                "tension_score": tension,
                "conflict_count": len(conflicts),
                "conflicts": [
                    {
                        "id": c.conflict_id,
                        "edge_a": c.edge_a_id,
                        "edge_b": c.edge_b_id,
                        "shared_vertices": c.shared_vertices,
                        "tension": c.tension_score,
                        "description": c.description,
                    }
                    for c in conflicts
                ],
            }

        elif name == "export_graph":
            fmt = args.get("format", "mermaid")
            if fmt == "mermaid":
                return {"status": "ok", "mermaid": export_to_mermaid(self.hg)}
            return {"status": "ok", "graph": self.hg.to_dict()}

        raise ValueError(f"Unknown tool: {name}")

    def run_stdio(self) -> None:
        """Runs the stdio JSON-RPC loop."""
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                req = json.loads(line)
                resp = self.handle_request(req)
                sys.stdout.write(json.dumps(resp) + "\n")
                sys.stdout.flush()
            except Exception as e:
                err_resp = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": str(e)}}
                sys.stdout.write(json.dumps(err_resp) + "\n")
                sys.stdout.flush()
