"""Bearer-token identity and grants for the REST surface.

Authentication and authorization used to stop at the MCP server, which left them as a
lock on one door of a building with two: anyone who could reach the REST port could send
messages as any agent, read any inbox, and read or write everything the grants were
withholding over MCP. This module applies the same policy, read from the same settings,
so turning on `WALD_REQUIRE_AUTH` / `WALD_ENFORCE_AUTHZ` means the same thing on both.

Tokens are the ones `wald-token` issues, sent as ``Authorization: Bearer <token>``. A
presented token that does not verify is rejected even with authentication off: a caller
that sent credentials expects to act as someone, and quietly demoting it to anonymous
would turn a typo into a request made as nobody.
"""

from __future__ import annotations

from dataclasses import dataclass

from fastapi import Depends, HTTPException, Request
from sqlalchemy.orm import Session

from wald.config import Settings, get_settings
from wald.db import get_session
from wald.models import Agent
from wald.services import authz, tokens


@dataclass(frozen=True)
class Caller:
    """Who is calling (None without a token) and what they may do."""

    agent: Agent | None
    grants: authz.Grants

    @property
    def slug(self) -> str | None:
        return self.agent.slug if self.agent else None


def _unauthorized(detail: str) -> HTTPException:
    return HTTPException(status_code=401, detail=detail, headers={"WWW-Authenticate": "Bearer"})


def _bearer(request: Request) -> str | None:
    header = request.headers.get("authorization")
    if not header:
        return None
    scheme, _, token = header.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise _unauthorized("expected 'Authorization: Bearer <token>'")
    return token.strip()


def get_caller(
    request: Request,
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_settings),
) -> Caller:
    token = _bearer(request)
    agent = tokens.resolve(session, token) if token else None
    if token and agent is None:
        raise _unauthorized("invalid or revoked token")
    # Grants attach to a proven identity, so enforcement implies authentication even if
    # the operator forgot the first switch -- the API also refuses to start that way.
    if agent is None and (settings.require_auth or settings.enforce_authz):
        raise _unauthorized("authentication required: present a bearer token")

    if not settings.enforce_authz:
        return Caller(agent, authz.ALLOW_ALL)
    try:
        return Caller(agent, authz.Grants(agent.grants or []))
    except ValueError as exc:
        # Same as MCP: a malformed grant denies, and says where to fix it.
        raise HTTPException(
            status_code=403, detail=f"agent '{agent.slug}' has an invalid grant: {exc}"
        ) from exc


def require(caller: Caller, pillar: str, action: str, selector: str) -> None:
    if not caller.grants.allows(pillar, action, selector):
        raise HTTPException(
            status_code=403,
            detail=f"not authorized: this agent has no '{pillar}:{action}:{selector}' grant",
        )
