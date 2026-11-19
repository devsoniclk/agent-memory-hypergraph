# Changelog

All notable changes to `agent-memory-hypergraph` are documented in this file.

## [0.1.0] - 2026-11-20

### Added
- Native higher-order directed and undirected hyperedge engine (`agent_hypergraph.models`).
- Hypergraph PageRank power iteration and random walk spectral diffusion (`agent_hypergraph.spectral`).
- Ebbinghaus temporal decay, access reinforcement, and episodic-to-semantic promotion (`agent_hypergraph.temporal`).
- H-hop neighborhood expansion and causal reachability chain search (`agent_hypergraph.retrieval`).
- Swarm assertion contradiction detection and hyperclique tension scoring (`agent_hypergraph.conflict`).
- FastMCP stdio JSON-RPC 2.0 protocol suite exposing 7 memory tools (`agent_hypergraph.mcp_server`).
- Click and Rich terminal command-line interface (`agent_hypergraph.cli`).
- Comprehensive test suite with 39 unit and integration tests and 93% coverage.
