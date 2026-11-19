import os
import tempfile
import pytest
from click.testing import CliRunner
from agent_hypergraph.cli import main


def test_cli_lifecycle():
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as td:
        gpath = os.path.join(td, "test_graph.json")

        # 1. Insert vertices
        res1 = runner.invoke(main, ["--graph-file", gpath, "insert-vertex", "AgentOne", "-t", "agent", "--id", "a1"])
        assert res1.exit_code == 0
        assert "Ingested Vertex" in res1.output

        res2 = runner.invoke(main, ["--graph-file", gpath, "insert-vertex", "KernelLog", "-t", "entity", "--id", "log1"])
        assert res2.exit_code == 0

        # 2. Insert hyperedge
        res3 = runner.invoke(
            main,
            [
                "--graph-file",
                gpath,
                "insert-edge",
                "TraceAnalysis",
                "-T",
                "a1",
                "-H",
                "log1",
                "-r",
                "causal",
            ],
        )
        assert res3.exit_code == 0
        assert "Ingested Hyperedge" in res3.output

        # 3. Query
        res4 = runner.invoke(main, ["--graph-file", gpath, "query", "a1", "--hops", "1"])
        assert res4.exit_code == 0
        assert "Retrieved Hypergraph Memory Context" in res4.output

        # 4. Rank
        res5 = runner.invoke(main, ["--graph-file", gpath, "rank"])
        assert res5.exit_code == 0
        assert "Hypergraph PageRank Centrality" in res5.output

        # 5. Consolidate
        res6 = runner.invoke(main, ["--graph-file", gpath, "consolidate"])
        assert res6.exit_code == 0
        assert "Memory Consolidation Complete" in res6.output

        # 6. Conflicts
        res7 = runner.invoke(main, ["--graph-file", gpath, "conflicts"])
        assert res7.exit_code == 0
        assert "Total Hyperclique Tension" in res7.output

        # 7. Export
        res8 = runner.invoke(main, ["--graph-file", gpath, "export", "--format", "mermaid"])
        assert res8.exit_code == 0
        assert "flowchart TD" in res8.output
