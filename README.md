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
| RAG synthesis    | Anthropic Claude, or any OpenAI-compatible `/v1` endpoint    |
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

# 3. Load some content
uv run wald-seed examples/acme

# 4. Run the API — web UI for humans at /, REST at /wiki, /resources, /search
uv run wald-api                # http://localhost:8000

# 5. Run the MCP server (agents)
uv run wald-mcp                                # stdio, for an agent that spawns Wald
uv run wald-mcp --transport streamable-http    # http://localhost:8091/mcp, for the fleet
```

Without an LLM (`WALD_LLM_BASE_URL` / `ANTHROPIC_API_KEY`) or embedder (`WALD_EMBED_BASE_URL` /
`VOYAGE_API_KEY`) configured, Wald runs in **dev mode**: embeddings use a deterministic local hash
and the `/ask` endpoint returns retrieved context without LLM synthesis.
This keeps the whole stack runnable end-to-end with zero external dependencies.

## Embeddings without an API key

The hash fallback is not semantic — it keeps the plumbing runnable, but the semantic arm of hybrid
search contributes little, and retrieval rests almost entirely on full-text matching. Point Wald at
any OpenAI-compatible `/v1` endpoint instead — a self-hosted embedding server, vLLM, TEI,
llama.cpp, or OpenAI — and it takes precedence over Voyage:

```bash
WALD_EMBED_BASE_URL=http://127.0.0.1:8082/v1
WALD_EMBED_MODEL=          # omit for single-model servers that reject an unknown model
WALD_EMBED_DIM=2048        # must match the model
```

`WALD_EMBED_DIM` fixes the vector column width when the table is created, so changing it means
recreating the `embedding` table and re-running `wald-seed`. That is cheap by design — the content
directory is the source of truth and the database is a derived index of it. A wrong dimension is
reported against the setting rather than surfacing as a pgvector error about expected dimensions.

## Synthesis without an API key

`/ask` synthesis can run on any OpenAI-compatible `/v1` chat endpoint instead of Claude — a
self-hosted llama.cpp server, vLLM, or a [Witchgrid](https://github.com/Schneewolf-Labs/Witchgrid)
routing URL such as `http://cp:8765/v1/llama/<profile>/v1`. When set, it takes precedence over
`ANTHROPIC_API_KEY`:

```bash
WALD_LLM_BASE_URL=http://127.0.0.1:8080/v1
WALD_LLM_MODEL=            # omit for single-model servers that reject an unknown model
WALD_LLM_API_KEY=          # optional bearer token
```

`<think>…</think>` blocks that reasoning models emit inline are stripped from the answer. A failing
endpoint is reported against `WALD_LLM_BASE_URL` rather than as a bare HTTP error.


The web UI is served at `/`: a faceted index of the wiki, resource directory and agent registry,
plus search. Wiki pages render at `/ui/wiki/<slug>`, resources at `/ui/resources/<slug>`, agents at
`/ui/agents/<slug>`.

Server-rendered, no build step and no JavaScript, so a wiki page is a real URL that opens when
someone pastes it into chat. It is a client of the same service layer the REST API and MCP server
use — there is no second implementation of "what is a search result".

Markdown is rendered with **raw HTML disabled**. That closes stored XSS through `POST /wiki`
(unauthenticated unless `WALD_REQUIRE_AUTH` is on), and it is also what makes the hub's own pages correct: several document literal
`<tool_call>` syntax that an HTML-aware parser would silently swallow.

## Agents

An MCP client points at Wald and gets thirteen tools: `search_wald`, `ask_wald`, `get_wiki_page`,
`write_wiki_page`, `list_resources`, `get_resource`, `find_agents`, `list_agents`,
`register_agent`, `send_agent_message`, `read_inbox`, `ack_messages`, `get_conversation`.

`write_wiki_page` is how an agent records a lesson in the hub instead of carrying it
alone. Writing identical content back is a no-op rather than a version bump, so page
history stays a record of actual edits — the same discipline as the seed loader.

`register_agent` is how a fleet discovers itself instead of every instance carrying a
hand-maintained list of every other instance — the N² configuration problem. An agent announces
its slug, capabilities and endpoint; peers resolve each other through `find_agents`. Re-registering
after a restart updates the same row.

A2A messaging is a mailbox. `read_inbox` marks queued messages **delivered**, not read; they stay
in the unread set until `ack_messages`. An agent that reads its inbox and then crashes therefore
sees the work again, because losing queued work silently is worse than delivering it twice.

Every message names its sender and recipient by slug (`from_agent`, `to_agent`), since slugs
are how agents address each other and the only identity they can check a message against.
With authentication off that sender is still just a claim; see below.

## Agent authentication

Each agent can hold a bearer token, which turns its identity from something it *asserts* into
something it *proves*:

```bash
uv run wald-token kira            # issue (printed once)
uv run wald-token kira --revoke   # disable access
```

Then enable it and point clients at the hub with the token:

```bash
WALD_REQUIRE_AUTH=true uv run wald-mcp --transport streamable-http
```

```toml
[[mcp.servers]]
name = "wald"
url = "http://wald.internal:8091/mcp"
headers = { Authorization = "Bearer $WALD_TOKEN" }
```

With it on, **`from_agent` is ignored** — `send_agent_message` sends as whoever the token
belongs to, and `read_inbox`/`ack_messages` operate on that agent's own mailbox. Honouring the
parameter as a fallback would reinstate exactly the spoof the token prevents, so it is dropped
rather than merely deprioritised. An unauthenticated request never reaches a tool at all.

Only the SHA-256 of a token is stored. The hub keeps the means to *check* an identity, not to
present one, so a database dump reveals which agents exist rather than how to impersonate them
— and a lost token can only be replaced, never recovered. Issuing again invalidates the
previous token, so rotation and revocation are the same operation and two live tokens for one
agent cannot coexist. Retiring an agent disables its token without a separate step.

> Authentication is **off by default**, so an existing hub does not lock out every agent the
> moment it upgrades. Until you turn it on, `from_agent` is still a claim, and `WALD_MCP_HOST`
> stays on loopback for that reason.

## Agent authorization

Authentication settles *who* is calling; grants settle *what* they may do. Each agent's
registry entry (its TOML file, or `register_agent`) can carry a `grants` list:

```toml
grants = ["wiki:read:*", "wiki:write:agent-notes", "resource:read:merlina"]
```

A grant is `pillar:action:selector` — wiki grants select a **space**, resource grants a
**slug**, `*` matches all, and the grammar is validated strictly at load time so a typo
fails the seed instead of surfacing later as a mystery denial.

Enforcement is a second opt-in on top of authentication:

```bash
WALD_REQUIRE_AUTH=true WALD_ENFORCE_AUTHZ=true uv run wald-mcp --transport streamable-http
```

With it on, `get_wiki_page` and `get_resource` check the caller's grants, `list_resources`
filters to what it may read, `write_wiki_page` requires `wiki:write` on the target space,
and — the part that matters most — `search_wald` and `ask_wald` **scope retrieval itself**:
context an agent may not read never reaches the synthesis prompt, so an answer cannot
become a paraphrase channel around a permission. Agent-registry entries stay visible to
everyone, because discovery is what the registry is for, and A2A mailboxes are already
bound to the verified token identity.

Authz without authn would be a lock on a door with no wall, so `WALD_ENFORCE_AUTHZ`
requires `WALD_REQUIRE_AUTH` and the server refuses the combination outright. An agent
with an empty grants list under enforcement has no knowledge access — deny by default.

### The REST API and web UI

Both settings apply to the REST API exactly as to MCP; a policy that held on one port and
not the other would not be a policy. Send the same token as `Authorization: Bearer <token>`:

- Without a valid token every route answers `401`, except `/health`. A token that does not
  verify is rejected even with authentication off, rather than treated as anonymous.
- `POST /agents/messages` sends as the token's agent and ignores `from_agent`;
  `/agents/{slug}/inbox` is readable only by its owner, `/agents/threads/{id}` only by the
  thread's participants, who are also the only ones who can post into it (on MCP too).
- `POST /agents` may only register the caller's own slug.
- Under enforcement, `/wiki`, `/resources`, `/search` and `/ask` apply the same grants as
  the MCP tools. What a caller may not read answers `404`, as if absent, so the API does
  not reveal which slugs exist. `POST /resources` needs `resource:write:<slug>`.

The web UI has no login — Wald has agent identities, not human ones — so while
authentication is on it is **closed** (`403`) rather than a way around it. Set
`WALD_WEB_UI_OPEN=true` if it sits behind authentication of your own, such as an SSO proxy
or VPN. `wald-api` binds to `127.0.0.1` by default for the same reason `wald-mcp` does; set
`WALD_HOST` once authentication is on.

## Background indexing

Writes no longer block on the embedding provider: a write commits first, and re-embedding
runs on a single worker thread that re-reads the committed row (`WALD_BACKGROUND_INDEXING`,
on by default). The tradeoff is a moment where a write has landed but search does not see
it yet; set it to `false` for read-your-writes search. The seed loader always indexes
inline, because `--dry-run` must count chunks and roll everything back in one transaction.
A failed background reindex logs and leaves the previous chunks standing, so search
degrades to slightly-stale rather than half-indexed.

## Content lives in files

The hub can be written through the REST API, but the intended source of truth is a directory of
plain files that `wald-seed` loads:

```
content/
  wiki/*.md          markdown with TOML frontmatter (+++ fenced)
  resources/*.toml   what a system is, how to connect, where its secret lives
  agents/*.toml      the agent registry
```

Slugs default to the filename stem. Loading is an idempotent upsert, and unchanged files are
skipped — re-running the loader does not manufacture wiki revisions, so page history stays a record
of what someone actually edited. `--dry-run` reports what would change and rolls back.

Keeping content in files means it can live wherever an organization already keeps private material,
review and history come from whatever version control wraps that directory, and rebuilding the hub
is one command — which is what makes the database safe to drop.

> **`content/` is gitignored, and that is deliberate.** A real hub holds internal process docs,
> hostnames, ports, and which vault path holds which credential. That is precisely what must not be
> committed to a public repo, so the ignore rule covers the whole directory: a new file dropped in
> is ignored by default rather than only if someone remembered to name it correctly. The committed
> `examples/acme` tenant is fictional.

Resource and agent entries record a **reference** to a secret (`vault://kv/...`), never the secret
itself. Wald tells an agent which door to knock on and which key to fetch; it is not the keyring.

## Status

This is an initial scaffold. Every pillar has a working data model, service layer, REST router, and
MCP tool, with clearly marked `TODO`s where the real implementation goes. See `ARCHITECTURE.md` →
"Roadmap".
