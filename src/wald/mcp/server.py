"""Wald MCP server.

The agent-facing front door. Any MCP-capable agent connects and gets tools to search the
hub, read wiki pages, look up how to connect to a resource, discover other agents, and
message them. These tools wrap the exact same service layer the REST API uses.

**Transport.** stdio suits an agent that spawns Wald as a child process, and that is all
the scaffold supported -- which quietly meant the hub could only serve agents on the same
machine, while the point of a company-wide hub is the ones that are not. `streamable-http`
serves the whole fleet from one endpoint. Set `WALD_MCP_TRANSPORT`, or pass `--transport`.

**Authentication: there is none yet, and it matters here.** `send_agent_message` takes the
sender's identity as an argument, so any caller can claim to be any agent. That is
survivable on a trusted network and is not survivable on an open one. Until agent-scoped
tokens land, an HTTP-served Wald belongs behind a network boundary, and `WALD_MCP_HOST`
defaults accordingly.
"""

from __future__ import annotations

import uuid
from typing import Any

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from sqlalchemy import select

from wald.config import get_settings
from wald.db import SchemaMismatch, SessionLocal, check_embedding_dim, engine
from wald.models import Agent, Resource, WikiPage
from wald.services import a2a, ingest, rag
from wald.services import search as search_svc

_settings = get_settings()

mcp = FastMCP("wald", host=_settings.mcp_host, port=_settings.mcp_port)


def _require_agent(session: Any, ref: str, role: str) -> Agent:
    """Resolve an agent or fail the tool call outright.

    Returning `{"error": ...}` from a tool is a success as far as MCP is concerned, so the
    calling agent is told the call worked and has to notice the payload disagrees. Raising
    sets `isError`, which every MCP client already routes to its failure path.
    """
    agent = a2a.resolve_agent(session, ref)
    if agent is None:
        known = session.scalars(select(Agent.slug).where(Agent.status == "active")).all()
        raise ToolError(
            f"{role} agent '{ref}' is not registered. "
            f"Registered agents: {', '.join(sorted(known)) or '(none)'}"
        )
    return agent


def _parse_uuid(value: str, label: str) -> uuid.UUID:
    try:
        return uuid.UUID(value)
    except ValueError as exc:
        raise ToolError(f"invalid {label}: {exc}") from exc


def _agent_summary(agent: Agent) -> dict[str, Any]:
    return {
        "slug": agent.slug,
        "name": agent.name,
        "description": agent.description,
        "capabilities": list(agent.capabilities),
        "protocol": agent.protocol,
        "endpoint_url": agent.endpoint_url,
        "status": agent.status,
    }


def _message_summary(msg: Any) -> dict[str, Any]:
    return {
        "id": str(msg.id),
        "thread_id": str(msg.thread_id),
        "from_agent_id": str(msg.from_agent_id),
        "role": msg.role,
        "content": msg.content,
        "status": msg.status,
        "created_at": msg.created_at.isoformat(),
    }


# --- Knowledge ------------------------------------------------------------
@mcp.tool()
def search_wald(query: str, top_k: int = 6) -> list[dict[str, Any]]:
    """Hybrid search across the wiki, resource directory, and agent registry."""
    with SessionLocal() as session:
        return [
            hit.model_dump(mode="json") for hit in search_svc.search(session, query, top_k=top_k)
        ]


@mcp.tool()
def ask_wald(question: str, top_k: int = 6) -> dict[str, Any]:
    """Ask a natural-language question; returns a synthesized, cited answer."""
    with SessionLocal() as session:
        return rag.ask(session, question, top_k=top_k).model_dump(mode="json")


@mcp.tool()
def get_wiki_page(slug: str) -> dict[str, Any]:
    """Fetch a wiki page by slug (returns title, markdown content, tags)."""
    with SessionLocal() as session:
        page = session.scalar(select(WikiPage).where(WikiPage.slug == slug))
        if page is None:
            # A bare null reads to a model as "this page exists and is empty". Naming the
            # alternatives turns a typo into a recoverable second attempt.
            known = session.scalars(select(WikiPage.slug).order_by(WikiPage.slug)).all()
            raise ToolError(f"no wiki page '{slug}'. Pages: {', '.join(known) or '(none)'}")
        return {
            "slug": page.slug,
            "title": page.title,
            "space": page.space,
            "content": page.content,
            "tags": list(page.tags),
            "version": page.version,
        }


# --- Resource directory ---------------------------------------------------
@mcp.tool()
def list_resources(kind: str | None = None) -> list[dict[str, Any]]:
    """List enterprise resources, optionally filtered by kind (database/api/service/tool/dataset)."""
    with SessionLocal() as session:
        stmt = select(Resource).order_by(Resource.name)
        if kind:
            stmt = stmt.where(Resource.kind == kind)
        return [
            {"slug": r.slug, "name": r.name, "kind": r.kind, "description": r.description}
            for r in session.scalars(stmt)
        ]


@mcp.tool()
def get_resource(slug: str) -> dict[str, Any]:
    """Get full details for one resource, including how to connect and authenticate.

    `auth` holds a *reference* to where a credential lives, never the credential itself.
    """
    with SessionLocal() as session:
        r = session.scalar(select(Resource).where(Resource.slug == slug))
        if r is None:
            known = session.scalars(select(Resource.slug).order_by(Resource.slug)).all()
            raise ToolError(f"no resource '{slug}'. Resources: {', '.join(known) or '(none)'}")
        return {
            "slug": r.slug,
            "name": r.name,
            "kind": r.kind,
            "description": r.description,
            "connection": r.connection,
            "auth": r.auth,
            "docs_url": r.docs_url,
            "status": r.status,
        }


# --- Agent directory ------------------------------------------------------
@mcp.tool()
def find_agents(capability: str) -> list[dict[str, Any]]:
    """Discover registered agents that advertise a given capability."""
    with SessionLocal() as session:
        return [_agent_summary(a) for a in a2a.find_agents_by_capability(session, capability)]


@mcp.tool()
def list_agents(include_inactive: bool = False) -> list[dict[str, Any]]:
    """List every agent in the registry — the fleet's peer directory."""
    with SessionLocal() as session:
        stmt = select(Agent).order_by(Agent.name)
        if not include_inactive:
            stmt = stmt.where(Agent.status == "active")
        return [_agent_summary(a) for a in session.scalars(stmt)]


@mcp.tool()
def register_agent(
    slug: str,
    name: str,
    description: str = "",
    capabilities: list[str] | None = None,
    endpoint_url: str | None = None,
    protocol: str = "mcp",
    owner: str | None = None,
) -> dict[str, Any]:
    """Announce this agent to the registry, or update its existing entry.

    Lets a fleet discover itself instead of every instance carrying a hand-maintained list
    of every other instance. Re-registering after a restart updates the same row.
    """
    with SessionLocal() as session:
        agent, created = a2a.register(
            session,
            {
                "slug": slug,
                "name": name,
                "description": description,
                "capabilities": capabilities or [],
                "endpoint_url": endpoint_url,
                "protocol": protocol,
                "owner": owner,
                "status": "active",
            },
        )
        ingest.index_agent(session, agent)
        session.commit()
        return {"registered": _agent_summary(agent), "created": created}


# --- A2A messaging --------------------------------------------------------
@mcp.tool()
def send_agent_message(
    from_agent: str,
    to_agent: str,
    content: str,
    role: str = "request",
    thread_id: str | None = None,
) -> dict[str, Any]:
    """Route a message from one registered agent to another (A2A).

    Pass `thread_id` to continue an existing exchange; omit it to start a new one.
    """
    with SessionLocal() as session:
        sender = _require_agent(session, from_agent, "sender")
        target = _require_agent(session, to_agent, "target")
        msg = a2a.send_message(
            session,
            from_agent=sender,
            to_agent=target,
            content=content,
            role=role,
            thread_id=_parse_uuid(thread_id, "thread_id") if thread_id else None,
        )
        session.commit()
        return {"message_id": str(msg.id), "thread_id": str(msg.thread_id), "status": msg.status}


@mcp.tool()
def read_inbox(agent: str, unread_only: bool = True, limit: int = 20) -> list[dict[str, Any]]:
    """Read messages addressed to an agent.

    Fetching marks queued messages as delivered but not as read: they stay in the unread
    set until `ack_messages`, so an agent that reads its inbox and then crashes sees the
    work again instead of losing it.
    """
    with SessionLocal() as session:
        target = _require_agent(session, agent, "recipient")
        messages = a2a.inbox(session, target, unread_only=unread_only)[:limit]
        a2a.mark_delivered(messages)
        out = [_message_summary(m) for m in messages]
        session.commit()
        return out


@mcp.tool()
def ack_messages(agent: str, message_ids: list[str]) -> dict[str, Any]:
    """Mark messages in this agent's inbox as read, once they have been acted on."""
    with SessionLocal() as session:
        target = _require_agent(session, agent, "recipient")
        ids = [_parse_uuid(m, "message id") for m in message_ids]
        marked = a2a.mark_read(session, target, ids)
        session.commit()
        return {"acknowledged": marked}


@mcp.tool()
def get_conversation(thread_id: str) -> list[dict[str, Any]]:
    """Read a whole A2A thread in order, oldest first."""
    with SessionLocal() as session:
        return [
            _message_summary(m) for m in a2a.thread(session, _parse_uuid(thread_id, "thread_id"))
        ]


def run() -> None:
    """Console-script entry point (``wald-mcp``)."""
    import argparse

    parser = argparse.ArgumentParser(description="Serve Wald over MCP.")
    parser.add_argument(
        "--transport",
        choices=("stdio", "streamable-http", "sse"),
        default=_settings.mcp_transport,
        help="stdio for a local child process; streamable-http for remote agents",
    )
    args = parser.parse_args()

    # Fail at startup rather than on every query. A server whose configured dimension no
    # longer matches the stored column answers `list_tools` perfectly and then fails every
    # search, which presents to the agent using it as "all your tools are broken" -- a much
    # harder thing to diagnose from the other end of an MCP connection than a server that
    # declined to start.
    try:
        check_embedding_dim(engine)
    except SchemaMismatch as exc:
        raise SystemExit(f"wald-mcp: {exc}") from exc

    if args.transport != "stdio":
        # stdio must keep stdout clean for the protocol itself, so this only prints when
        # stdout is not the transport.
        print(
            f"Wald MCP on {args.transport} at http://{_settings.mcp_host}:{_settings.mcp_port}/mcp",
            flush=True,
        )
    mcp.run(transport=args.transport)


if __name__ == "__main__":
    run()
