# Wald — Architecture

## Guiding principle: agent-first, human-friendly

Most knowledge bases are built for humans and bolted onto agents later. Wald inverts that. Every
capability is exposed as a clean programmatic surface first (REST + MCP), and the human web UI is a
client of that same surface. The payoff: anything a person can do, an agent can do, and vice versa —
no second-class API.

The connective tissue is the **MCP server**. An enterprise agent points its MCP client at Wald and
immediately gets tools to search the wiki, look up how to connect to a resource, find another agent,
or ask a natural-language question over everything. The same core service powers the web UI.

## Components

```
apps/
  Core API (FastAPI)   — REST for humans + UI; owns the domain logic
  MCP server           — exposes the domain to agents as MCP tools
  A2A router           — agent-to-agent message routing (lives in the core service)

data/
  PostgreSQL           — relational store for all four pillars
  pgvector             — embeddings column for semantic search, same database
```

Single-process-friendly for development; each surface can be split into its own deployable later.

## Why these choices

- **Python** — the strongest ecosystem for RAG, embeddings, and agent tooling, with first-class
  official SDKs for both MCP and Anthropic.
- **Postgres + pgvector** — one datastore for relational data, full-text search, *and* vectors.
  Hybrid search is a single SQL query; there's no second system to operate. We can graduate the
  vector workload to a dedicated store (Qdrant/Weaviate) later without changing the domain model.
- **MCP** — the emerging standard for connecting agents to tools/data. Exposing Wald as an MCP
  server means any MCP-capable agent (Claude, or anything else) can consume it with no bespoke glue.
- **Claude `claude-opus-4-8` + Voyage `voyage-3.5`** — Claude for answer synthesis, Voyage for
  embeddings (Anthropic's recommended embedding provider). Both are swappable behind a provider
  interface, and both degrade gracefully to a keyless dev mode.

## Data model

All tables share a UUID primary key and `created_at` / `updated_at`.

### Wiki
- `wiki_page` — `slug`, `title`, `space`, `content` (markdown), `tags[]`, `version`.
- `wiki_page_revision` — immutable history (`page_id`, `version`, `title`, `content`, `author`).

### Resource directory
- `resource` — `slug`, `name`, `kind` (database | api | service | tool | dataset), `description`,
  `connection` (JSON: endpoints/hosts/params), `auth` (JSON: method + secret references — **never
  raw secrets**), `docs_url`, `owner`, `tags[]`, `status`.

### Agent directory + A2A
- `agent` — `slug`, `name`, `description`, `capabilities[]`, `endpoint_url`, `protocol`
  (mcp | a2a | http), `auth` (JSON), `owner`, `status`, `agent_card` (JSON: full descriptor).
- `agent_message` — `thread_id`, `from_agent_id`, `to_agent_id`, `role`, `content`, `status`
  (queued | delivered | read | failed). The minimal substrate for routing messages between agents.

### RAG
- `embedding` — unified index across pillars: `source_type` (wiki | resource | agent),
  `source_id`, `chunk_index`, `content` (text), `embedding` (vector), `meta` (JSON). One table so
  search spans everything in a single query.

## Retrieval

`search` does **hybrid** retrieval:
1. **Lexical** — Postgres full-text / trigram over chunk text.
2. **Semantic** — cosine distance over `embedding` via pgvector.
3. **Fuse** — reciprocal-rank fusion of the two result sets.

`ask` (RAG) runs `search`, then hands the top chunks to Claude to synthesize a cited answer.

## A2A model (v1)

Agents register a card. Discovery is "find an agent by capability". Messaging is a store-and-route
mailbox: an agent (or a human) posts a message addressed to another agent; the target agent polls or
is pushed its inbox. This is intentionally transport-light to start — it can grow into
streaming/long-running conversations and richer protocols (e.g. aligning with emerging A2A specs).

## Roadmap

- **Done:** data models, service layer, REST + MCP surfaces, dev-mode fallbacks; file-backed
  content loading (`wald-seed`); MCP over stdio *and* streamable-http; the A2A round trip
  (register / discover / send / inbox / ack); hybrid search on Postgres full-text + pgvector
  fused by RRF, with a functional GIN index; the human web UI, server-rendered at `/`.
  Agent authentication: bearer tokens (SHA-256 stored, never the token), opt-in via
  `WALD_REQUIRE_AUTH`, with `from_agent` derived from the verified identity rather than
  accepted as a parameter.
- **Next:** **authz** — authentication says *who* is calling; nothing yet says *what* they may
  do. Every authenticated agent can read every resource and write any wiki page, so per-space
  and per-resource permissions are the next piece, and they are what the web UI needs before it
  can offer editing. The REST surface is also still unauthenticated, which is why it and the
  MCP surface should not be exposed on the same terms. Then Alembic migrations; background
  re-embedding on writes (ingestion is inline and synchronous, so a wiki write blocks on an
  embedding API call).
- **Later:** push-based A2A (webhooks/streaming); per-space permissions; audit log; connectors that
  auto-populate the resource directory; eval harness for RAG answer quality.
