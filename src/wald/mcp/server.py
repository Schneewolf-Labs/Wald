"""Wald MCP server.

The agent-facing front door. Any MCP-capable agent connects and gets tools to search the
hub, read wiki pages, look up how to connect to a resource, discover other agents, and
message them. These tools wrap the exact same service layer the REST API uses.

**Transport.** stdio suits an agent that spawns Wald as a child process, and that is all
the scaffold supported -- which quietly meant the hub could only serve agents on the same
machine, while the point of a company-wide hub is the ones that are not. `streamable-http`
serves the whole fleet from one endpoint. Set `WALD_MCP_TRANSPORT`, or pass `--transport`.

**Authentication** is opt-in (`WALD_REQUIRE_AUTH`, see mcp/auth.py): with it on, identity
comes from the bearer token and `from_agent` is ignored. **Authorization** is a second
opt-in on top (`WALD_ENFORCE_AUTHZ`, see services/authz.py): grants on the agent's
registry entry decide which wiki spaces it may read or write and which resources it may
see. With both off -- the default -- identity is a claim and every caller sees everything,
which is why `WALD_MCP_HOST` stays on loopback until someone decides otherwise.
"""

from __future__ import annotations

import uuid
from typing import Any

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from sqlalchemy import select

from wald.config import get_settings
from wald.db import SchemaMismatch, SessionLocal, check_embedding_dim, engine
from wald.mcp.auth import WaldTokenVerifier, authenticated_slug, caller_or
from wald.models import Agent, Resource, WikiPage
from wald.services import a2a, authz, background, rag
from wald.services import search as search_svc
from wald.services import wiki as wiki_svc

_settings = get_settings()

# Auth is wired in only when enabled, because passing a token_verifier makes the SDK reject
# every unauthenticated request -- which is the point when it is on, and a lockout when a hub
# has upgraded but not yet issued any tokens.
if _settings.require_auth:
    from mcp.server.auth.settings import AuthSettings

    mcp = FastMCP(
        "wald",
        host=_settings.mcp_host,
        port=_settings.mcp_port,
        token_verifier=WaldTokenVerifier(),
        auth=AuthSettings(
            issuer_url=_settings.auth_issuer_url,
            resource_server_url=_settings.auth_issuer_url,
            required_scopes=["agent"],
        ),
    )
else:
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


def _caller(claimed: str | None) -> str:
    """Which agent this call is from -- the verified token if there is one, else the claim."""
    try:
        return caller_or(claimed, require=_settings.require_auth)
    except (PermissionError, ValueError) as exc:
        # ToolError so the caller's MCP client routes it to its failure path rather than
        # being handed a successful-looking result it has to inspect.
        raise ToolError(str(exc)) from exc


def _caller_grants(session: Any) -> authz.Grants:
    """The caller's grant set, or ALLOW_ALL while enforcement is off.

    With enforcement on there is always a verified identity behind the request (the
    server refuses to start with authz on and auth off), and that identity always maps to
    a registered agent, because the token that proved it is a column on the agent's row.
    """
    if not _settings.enforce_authz:
        return authz.ALLOW_ALL
    slug = authenticated_slug()
    agent = a2a.resolve_agent(session, slug) if slug else None
    if agent is None:
        raise ToolError("authorization is enforced and this request carries no verified agent")
    try:
        return authz.Grants(agent.grants or [])
    except ValueError as exc:
        # A malformed grant in the registry denies rather than allows, but says why, so
        # the fix lands in the agent's TOML file instead of in a debugging session.
        raise ToolError(f"agent '{agent.slug}' has an invalid grant: {exc}") from exc


def _require(grants: authz.Grants, pillar: str, action: str, selector: str) -> None:
    if not grants.allows(pillar, action, selector):
        raise ToolError(f"not authorized: this agent has no '{pillar}:{action}:{selector}' grant")


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
        "from_agent": msg.from_agent,
        "to_agent": msg.to_agent,
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
        grants = _caller_grants(session)
        return [
            hit.model_dump(mode="json")
            for hit in search_svc.search(session, query, top_k=top_k, grants=grants)
        ]


@mcp.tool()
def ask_wald(question: str, top_k: int = 6) -> dict[str, Any]:
    """Ask a natural-language question; returns a synthesized, cited answer."""
    with SessionLocal() as session:
        grants = _caller_grants(session)
        return rag.ask(session, question, top_k=top_k, grants=grants).model_dump(mode="json")


@mcp.tool()
def get_wiki_page(slug: str) -> dict[str, Any]:
    """Fetch a wiki page by slug (returns title, markdown content, tags)."""
    with SessionLocal() as session:
        grants = _caller_grants(session)
        page = session.scalar(select(WikiPage).where(WikiPage.slug == slug))
        if page is None:
            # A bare null reads to a model as "this page exists and is empty". Naming the
            # alternatives turns a typo into a recoverable second attempt -- scoped to what
            # this caller may read, so the hint is not also an inventory of what it may not.
            known = [
                s
                for s, sp in session.execute(
                    select(WikiPage.slug, WikiPage.space).order_by(WikiPage.slug)
                )
                if grants.allows("wiki", "read", sp)
            ]
            raise ToolError(f"no wiki page '{slug}'. Pages: {', '.join(known) or '(none)'}")
        _require(grants, "wiki", "read", page.space)
        return {
            "slug": page.slug,
            "title": page.title,
            "space": page.space,
            "content": page.content,
            "tags": list(page.tags),
            "version": page.version,
        }


@mcp.tool()
def write_wiki_page(
    slug: str,
    title: str,
    content: str,
    space: str | None = None,
    tags: list[str] | None = None,
    author: str | None = None,
) -> dict[str, Any]:
    """Create a wiki page, or update it (markdown; a new version is recorded on change).

    This is how an agent writes a lesson back into the hub instead of carrying it alone.
    Re-writing identical content is a no-op rather than a version bump. `space` and `tags`
    are kept as-is on update when omitted. The recorded author is the verified identity
    when there is one; `author` is only honoured without authentication.
    """
    with SessionLocal() as session:
        grants = _caller_grants(session)
        existing = session.scalar(select(WikiPage).where(WikiPage.slug == slug))
        # Both ends of a move need the grant: writing *into* a space, and rewriting a page
        # that currently lives in one.
        if existing is not None:
            _require(grants, "wiki", "write", existing.space)
        _require(grants, "wiki", "write", space or (existing.space if existing else "general"))
        page, created = wiki_svc.upsert_page(
            session,
            slug=slug,
            title=title,
            content=content,
            space=space,
            tags=tags,
            author=authenticated_slug() or author or "mcp",
        )
        result = {
            "slug": page.slug,
            "space": page.space,
            "version": page.version,
            "created": created,
        }
        background.finish_write(session, "wiki", page)
        return result


# --- Resource directory ---------------------------------------------------
@mcp.tool()
def list_resources(kind: str | None = None) -> list[dict[str, Any]]:
    """List enterprise resources, optionally filtered by kind (database/api/service/tool/dataset)."""
    with SessionLocal() as session:
        grants = _caller_grants(session)
        stmt = select(Resource).order_by(Resource.name)
        if kind:
            stmt = stmt.where(Resource.kind == kind)
        return [
            {"slug": r.slug, "name": r.name, "kind": r.kind, "description": r.description}
            for r in session.scalars(stmt)
            if grants.allows("resource", "read", r.slug)
        ]


@mcp.tool()
def get_resource(slug: str) -> dict[str, Any]:
    """Get full details for one resource, including how to connect and authenticate.

    `auth` holds a *reference* to where a credential lives, never the credential itself.
    """
    with SessionLocal() as session:
        grants = _caller_grants(session)
        r = session.scalar(select(Resource).where(Resource.slug == slug))
        if r is None:
            known = [
                s
                for s in session.scalars(select(Resource.slug).order_by(Resource.slug))
                if grants.allows("resource", "read", s)
            ]
            raise ToolError(f"no resource '{slug}'. Resources: {', '.join(known) or '(none)'}")
        _require(grants, "resource", "read", r.slug)
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

    With authentication on, an agent may only register itself. Peers resolve each other's
    `endpoint_url` from this row and send their own credentials there, so letting any token
    holder rewrite another agent's row hands it every message -- and token -- meant for that
    agent. It would also let anyone revive an agent an operator had retired.
    """
    verified = authenticated_slug()
    if verified and verified != slug:
        raise ToolError(f"authenticated as '{verified}'; an agent may only register itself")
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
        summary = _agent_summary(agent)
        background.finish_write(session, "agent", agent)
        return {"registered": summary, "created": created}


# --- A2A messaging --------------------------------------------------------
@mcp.tool()
def send_agent_message(
    to_agent: str,
    content: str,
    role: str = "request",
    thread_id: str | None = None,
    from_agent: str | None = None,
) -> dict[str, Any]:
    """Route a message from one registered agent to another (A2A).

    The sender is taken from the bearer token when authentication is enabled; `from_agent` is
    then ignored, because honouring it would reinstate exactly the spoof the token prevents.
    Without authentication it is required, and it is a claim rather than a proof.

    Pass `thread_id` to continue an existing exchange; omit it to start a new one.
    """
    with SessionLocal() as session:
        sender = _require_agent(session, _caller(from_agent), "sender")
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
def read_inbox(
    unread_only: bool = True, limit: int = 20, agent: str | None = None
) -> list[dict[str, Any]]:
    """Read messages addressed to this agent.

    Reads your *own* inbox: with authentication on, the mailbox is the one the token belongs
    to and `agent` is ignored. Otherwise anyone could read anyone's mail simply by naming it.

    Fetching marks queued messages as delivered but not as read: they stay in the unread
    set until `ack_messages`, so an agent that reads its inbox and then crashes sees the
    work again instead of losing it.
    """
    with SessionLocal() as session:
        target = _require_agent(session, _caller(agent), "recipient")
        messages = a2a.inbox(session, target, unread_only=unread_only)[:limit]
        a2a.mark_delivered(messages)
        out = [_message_summary(m) for m in messages]
        session.commit()
        return out


@mcp.tool()
def ack_messages(message_ids: list[str], agent: str | None = None) -> dict[str, Any]:
    """Mark messages in this agent's own inbox as read, once they have been acted on."""
    with SessionLocal() as session:
        target = _require_agent(session, _caller(agent), "recipient")
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

    # Grants attach to identities the token layer has *proved*; enforcing them against
    # claimed identities would be a lock on a door with no wall, so refuse the combination
    # outright rather than serving something that looks locked and is not.
    if _settings.enforce_authz and not _settings.require_auth:
        raise SystemExit(
            "wald-mcp: WALD_ENFORCE_AUTHZ requires WALD_REQUIRE_AUTH -- authorization "
            "without authentication would enforce grants against unverified identities"
        )

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
