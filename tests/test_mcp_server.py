import json
import pytest
from agent_hypergraph.mcp_server import AgentHypergraphMCPServer
from agent_hypergraph.models import Hypergraph


def test_mcp_initialize():
    server = AgentHypergraphMCPServer()
    req = {"jsonrpc": "2.0", "id": 1, "method": "initialize"}
    resp = server.handle_request(req)

    assert resp["id"] == 1
    assert resp["result"]["serverInfo"]["name"] == "agent-memory-hypergraph"


def test_mcp_tools_list():
    server = AgentHypergraphMCPServer()
    req = {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}
    resp = server.handle_request(req)

    tools = resp["result"]["tools"]
    tool_names = [t["name"] for t in tools]
    assert "add_vertex" in tool_names
    assert "add_hyperedge" in tool_names
    assert "query_subgraph" in tool_names
    assert "compute_pagerank" in tool_names
    assert "consolidate_memory" in tool_names
    assert "detect_conflicts" in tool_names
    assert "export_graph" in tool_names


def test_mcp_tool_execution_flow():
    server = AgentHypergraphMCPServer()

    # 1. Add vertex
    req_v = {
        "jsonrpc": "2.0",
        "id": 10,
        "method": "tools/call",
        "params": {
            "name": "add_vertex",
            "arguments": {"label": "Nova Agent", "vertex_type": "agent", "vertex_id": "agent_nova"},
        },
    }
    resp_v = server.handle_request(req_v)
    assert resp_v["id"] == 10
    v_content = json.loads(resp_v["result"]["content"][0]["text"])
    assert v_content["status"] == "ok"
    assert v_content["vertex"]["id"] == "agent_nova"

    # 2. Add hyperedge
    req_e = {
        "jsonrpc": "2.0",
        "id": 11,
        "method": "tools/call",
        "params": {
            "name": "add_hyperedge",
            "arguments": {
                "label": "Audit Tool Interaction",
                "tail": ["agent_nova", "tool_read"],
                "head": ["vuln_report"],
                "relation_type": "causal",
            },
        },
    }
    resp_e = server.handle_request(req_e)
    e_content = json.loads(resp_e["result"]["content"][0]["text"])
    assert e_content["status"] == "ok"

    # 3. Query subgraph
    req_q = {
        "jsonrpc": "2.0",
        "id": 12,
        "method": "tools/call",
        "params": {
            "name": "query_subgraph",
            "arguments": {"seed_vertices": ["agent_nova"], "max_hops": 2},
        },
    }
    resp_q = server.handle_request(req_q)
    q_content = json.loads(resp_q["result"]["content"][0]["text"])
    assert q_content["status"] == "ok"
    assert "prompt_context" in q_content

    # 4. PageRank
    req_pr = {
        "jsonrpc": "2.0",
        "id": 13,
        "method": "tools/call",
        "params": {"name": "compute_pagerank", "arguments": {}},
    }
    resp_pr = server.handle_request(req_pr)
    pr_content = json.loads(resp_pr["result"]["content"][0]["text"])
    assert pr_content["status"] == "ok"
    assert "agent_nova" in pr_content["pagerank"]

    # 5. Export graph
    req_exp = {
        "jsonrpc": "2.0",
        "id": 14,
        "method": "tools/call",
        "params": {"name": "export_graph", "arguments": {"format": "mermaid"}},
    }
    resp_exp = server.handle_request(req_exp)
    exp_content = json.loads(resp_exp["result"]["content"][0]["text"])
    assert exp_content["status"] == "ok"
    assert "flowchart TD" in exp_content["mermaid"]


def test_mcp_unknown_method_and_tool():
    server = AgentHypergraphMCPServer()
    resp_m = server.handle_request({"jsonrpc": "2.0", "id": 99, "method": "invalid/method"})
    assert "error" in resp_m
    assert resp_m["error"]["code"] == -32601

    resp_t = server.handle_request({
        "jsonrpc": "2.0",
        "id": 100,
        "method": "tools/call",
        "params": {"name": "unknown_tool", "arguments": {}},
    })
    assert "error" in resp_t
    assert resp_t["error"]["code"] == -32603
