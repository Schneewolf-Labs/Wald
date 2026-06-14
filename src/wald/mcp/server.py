"""Wald MCP server.

The agent-facing front door. Any MCP-capable agent can connect and get tools to
search the hub, read wiki pages, look up how to connect to a resource, discover
other agents, and message them. These tools wrap the exact same service layer the
REST API uses.
"""

from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP
from sqlalchemy import select

from wald.db import SessionLocal
from wald.models import Resource, WikiPage
from wald.services import a2a, rag
from wald.services import search as search_svc

mcp = FastMCP("wald")


@mcp.tool()
def search_wald(query: str, top_k: int = 6) -> list[dict[str, Any]]:
    """Hybrid search across the wiki, resource directory, and agent registry."""
    with SessionLocal() as session:
        return [hit.model_dump(mode="json") for hit in search_svc.search(session, query, top_k=top_k)]


@mcp.tool()
def ask_wald(question: str, top_k: int = 6) -> dict[str, Any]:
    """Ask a natural-language question; returns a synthesized, cited answer."""
    with SessionLocal() as session:
        return rag.ask(session, question, top_k=top_k).model_dump(mode="json")


@mcp.tool()
def get_wiki_page(slug: str) -> dict[str, Any] | None:
    """Fetch a wiki page by slug (returns title, markdown content, tags)."""
    with SessionLocal() as session:
        page = session.scalar(select(WikiPage).where(WikiPage.slug == slug))
        if page is None:
            return None
        return {
            "slug": page.slug,
            "title": page.title,
            "space": page.space,
            "content": page.content,
            "tags": list(page.tags),
            "version": page.version,
        }


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
def get_resource(slug: str) -> dict[str, Any] | None:
    """Get full details for one resource, including how to connect and authenticate."""
    with SessionLocal() as session:
        r = session.scalar(select(Resource).where(Resource.slug == slug))
        if r is None:
            return None
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


@mcp.tool()
def find_agents(capability: str) -> list[dict[str, Any]]:
    """Discover registered agents that advertise a given capability."""
    with SessionLocal() as session:
        return [
            {
                "slug": a.slug,
                "name": a.name,
                "capabilities": list(a.capabilities),
                "protocol": a.protocol,
                "endpoint_url": a.endpoint_url,
            }
            for a in a2a.find_agents_by_capability(session, capability)
        ]


@mcp.tool()
def send_agent_message(from_agent: str, to_agent: str, content: str, role: str = "request") -> dict[str, Any]:
    """Route a message from one registered agent to another (A2A)."""
    with SessionLocal() as session:
        sender = a2a.resolve_agent(session, from_agent)
        target = a2a.resolve_agent(session, to_agent)
        if sender is None or target is None:
            return {"error": "sender or target agent not found"}
        msg = a2a.send_message(
            session, from_agent=sender, to_agent=target, content=content, role=role
        )
        session.commit()
        return {"message_id": str(msg.id), "thread_id": str(msg.thread_id), "status": msg.status}


def run() -> None:
    """Console-script entry point (``wald-mcp``). Serves over stdio."""
    mcp.run()


if __name__ == "__main__":
    run()
