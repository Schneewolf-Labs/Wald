# Wald

> A forest of knowledge. The enterprise information hub for AI agents — and the humans who work with them.

Wald is a central place to collect, organize, and serve enterprise information so that
**both AI agents and people** can find what they need and act on it. It has two front doors:
a web UI for humans and a programmatic surface (REST + an **MCP server**) for agents.

## The four pillars

1. **Wiki** — versioned markdown pages about the company, products, processes. Human-editable, agent-readable.
2. **Resource directory** — a structured catalog of enterprise systems, datasources, and tools:
   what they are, how to connect, how to authenticate, how to use them.
3. **Agent directory + A2A** — a registry of the AI agents operating in the company (capabilities,
   endpoints, owners) plus a message-routing layer so agents can discover and talk to each other.
4. **RAG + search** — unified hybrid (keyword + semantic) retrieval across everything above, with
   an LLM-synthesized answer endpoint.

```
            ┌─────────── Web UI (humans) ──────────┐
            │                                       │
  Agents ──►│  MCP server  ──►  Core API  ──►  Postgres (+ pgvector)
            │                      │                │
            └──── A2A router ──────┘ ──► RAG / embeddings (Claude + Voyage)
```

## Tech stack

| Layer            | Choice                                                        |
| ---------------- | ------------------------------------------------------------ |
| Language         | Python 3.12                                                  |
| API              | FastAPI + Uvicorn                                            |
| Data / ORM       | PostgreSQL + `pgvector`, SQLAlchemy 2.0                      |
| Agent interface  | Model Context Protocol (MCP) server via the `mcp` SDK        |
| RAG synthesis    | Anthropic Claude (`claude-opus-4-8`)                         |
| Embeddings       | Voyage AI (`voyage-3.5`, 1024-dim) + keyless dev fallback    |
| Config           | `pydantic-settings` (env-driven)                            |
| Packaging        | `uv`                                                         |

See [`ARCHITECTURE.md`](./ARCHITECTURE.md) for the design rationale and data model.

## Quickstart

```bash
# 1. Start Postgres + pgvector
docker compose up -d

# 2. Install deps (uv) and create tables
uv sync
cp .env.example .env          # fill in keys when you have them
uv run wald-init-db

# 3. Run the API (humans + REST)
uv run wald-api                # http://localhost:8000/docs

# 4. Run the MCP server (agents)
uv run wald-mcp
```

Without `ANTHROPIC_API_KEY` / `VOYAGE_API_KEY` set, Wald runs in **dev mode**: embeddings use a
deterministic local hash and the `/ask` endpoint returns retrieved context without LLM synthesis.
This keeps the whole stack runnable end-to-end with zero external dependencies.

## Status

This is an initial scaffold. Every pillar has a working data model, service layer, REST router, and
MCP tool, with clearly marked `TODO`s where the real implementation goes. See `ARCHITECTURE.md` →
"Roadmap".
