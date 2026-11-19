from __future__ import annotations

import json
import sys
from typing import List, Optional

import click
from rich.console import Console
from rich.table import Table

from agent_hypergraph.conflict import ConflictDetector
from agent_hypergraph.mcp_server import AgentHypergraphMCPServer
from agent_hypergraph.models import Hypergraph, RelationType, VertexType
from agent_hypergraph.retrieval import HypergraphRetriever
from agent_hypergraph.spectral import compute_hypergraph_pagerank
from agent_hypergraph.storage import export_to_mermaid, load_from_json, save_to_json
from agent_hypergraph.temporal import TemporalConsolidator

console = Console()


@click.group()
@click.option("--graph-file", "-f", default="hypergraph.json", help="Path to persisted hypergraph JSON file.")
@click.pass_context
def main(ctx: click.Context, graph_file: str) -> None:
    """Agent Memory Hypergraph CLI: Multi-way relational memory for AI agent swarms."""
    ctx.ensure_object(dict)
    ctx.obj["graph_file"] = graph_file
    try:
        ctx.obj["hg"] = load_from_json(graph_file)
    except Exception:
        ctx.obj["hg"] = Hypergraph(name="agent_swarm_memory")


@main.command(name="insert-vertex")
@click.argument("label")
@click.option("--type", "-t", "vtype", default="concept", type=click.Choice([v.value for v in VertexType]))
@click.option("--id", "vid", default=None, help="Optional explicit vertex ID.")
@click.pass_context
def insert_vertex_cmd(ctx: click.Context, label: str, vtype: str, vid: Optional[str]) -> None:
    """Add a vertex (entity/concept/tool/agent) to the hypergraph."""
    hg: Hypergraph = ctx.obj["hg"]
    v = hg.add_vertex(label=label, vertex_type=VertexType(vtype), vertex_id=vid)
    save_to_json(hg, ctx.obj["graph_file"])
    console.print(f"[bold green]✓ Ingested Vertex:[/bold green] [cyan]{v.id}[/cyan] ({v.label}) [{v.vertex_type.value}]")


@main.command(name="insert-edge")
@click.argument("label")
@click.option("--tail", "-T", multiple=True, required=True, help="Tail/input vertex IDs.")
@click.option("--head", "-H", multiple=True, help="Head/output vertex IDs.")
@click.option("--relation", "-r", default="assertion", type=click.Choice([r.value for r in RelationType]))
@click.option("--confidence", "-c", default=1.0, type=float, help="Confidence [0.0, 1.0].")
@click.option("--agent", "-a", default=None, help="Source agent ID.")
@click.pass_context
def insert_edge_cmd(
    ctx: click.Context,
    label: str,
    tail: List[str],
    head: List[str],
    relation: str,
    confidence: float,
    agent: Optional[str],
) -> None:
    """Add an n-ary hyperedge connecting multiple vertices simultaneously."""
    hg: Hypergraph = ctx.obj["hg"]
    e = hg.add_hyperedge(
        label=label,
        tail=set(tail),
        head=set(head),
        relation_type=RelationType(relation),
        confidence=confidence,
        source_agent=agent,
    )
    save_to_json(hg, ctx.obj["graph_file"])
    console.print(
        f"[bold green]✓ Ingested Hyperedge:[/bold green] [magenta]{e.id}[/magenta] '{e.label}' "
        f"connecting {len(e.members)} vertices."
    )


@main.command(name="query")
@click.argument("seeds", nargs=-1, required=True)
@click.option("--hops", default=2, type=int, help="Max neighborhood expansion hops.")
@click.pass_context
def query_cmd(ctx: click.Context, seeds: List[str], hops: int) -> None:
    """Query multi-hop neighborhood around seed vertices."""
    hg: Hypergraph = ctx.obj["hg"]
    retriever = HypergraphRetriever(default_max_hops=hops)
    verts, edges = retriever.expand_subgraph(hg, list(seeds), max_hops=hops)
    save_to_json(hg, ctx.obj["graph_file"])

    formatted = retriever.format_for_prompt(verts, edges)
    console.print(formatted)


@main.command(name="rank")
@click.option("--damping", default=0.85, type=float, help="PageRank damping factor.")
@click.pass_context
def rank_cmd(ctx: click.Context, damping: float) -> None:
    """Compute Hypergraph PageRank centrality scores across all vertices."""
    hg: Hypergraph = ctx.obj["hg"]
    ranks = compute_hypergraph_pagerank(hg, damping=damping)

    table = Table(title="Hypergraph PageRank Centrality")
    table.add_column("Rank", style="cyan")
    table.add_column("Vertex ID", style="magenta")
    table.add_column("Label", style="green")
    table.add_column("Score", style="yellow")

    for i, (vid, score) in enumerate(ranks.items(), 1):
        v = hg.get_vertex(vid)
        lbl = v.label if v else ""
        table.add_row(str(i), vid, lbl, f"{score:.6f}")

    console.print(table)


@main.command(name="consolidate")
@click.pass_context
def consolidate_cmd(ctx: click.Context) -> None:
    """Run Ebbinghaus temporal consolidation, decay weights, and prune dead memory."""
    hg: Hypergraph = ctx.obj["hg"]
    consolidator = TemporalConsolidator()
    res = consolidator.consolidate(hg)
    save_to_json(hg, ctx.obj["graph_file"])
    console.print(f"[bold green]✓ Memory Consolidation Complete:[/bold green] {json.dumps(res, indent=2)}")


@main.command(name="conflicts")
@click.pass_context
def conflicts_cmd(ctx: click.Context) -> None:
    """Detect contradictory assertions and calculate hyperclique tension."""
    hg: Hypergraph = ctx.obj["hg"]
    detector = ConflictDetector()
    conflicts = detector.detect_conflicts(hg)
    tension = detector.compute_hyperclique_tension(hg)

    console.print(f"[bold yellow]Total Hyperclique Tension:[/bold yellow] {tension:.4f}")
    console.print(f"[bold yellow]Detected Conflicts:[/bold yellow] {len(conflicts)}")
    for c in conflicts:
        console.print(f"  • [red]{c.conflict_id}[/red]: {c.description} (tension: {c.tension_score:.2f})")


@main.command(name="export")
@click.option("--format", "-F", "fmt", default="mermaid", type=click.Choice(["mermaid", "json"]))
@click.pass_context
def export_cmd(ctx: click.Context, fmt: str) -> None:
    """Export current hypergraph state to Mermaid diagram or JSON."""
    hg: Hypergraph = ctx.obj["hg"]
    if fmt == "mermaid":
        console.print(export_to_mermaid(hg))
    else:
        console.print(json.dumps(hg.to_dict(), indent=2))


@main.command(name="serve")
@click.pass_context
def serve_cmd(ctx: click.Context) -> None:
    """Launch Model Context Protocol (MCP) stdio JSON-RPC server."""
    hg: Hypergraph = ctx.obj["hg"]
    server = AgentHypergraphMCPServer(hg=hg)
    server.run_stdio()


if __name__ == "__main__":
    main()
