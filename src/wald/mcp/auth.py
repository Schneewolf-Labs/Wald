"""Bearer-token verification for the MCP surface.

FastMCP takes a `TokenVerifier`, calls it once per request, and puts the resulting
`AccessToken` where a tool can read it with `get_access_token()`. That is the whole hook
needed to turn `from_agent` from a claim into a proof.

Authentication is **opt-in** (`WALD_REQUIRE_AUTH`). A hub already running on a trusted
network should not break because it upgraded, and a hub with no tokens issued yet would
otherwise lock out every agent at once. When it is off, tools behave exactly as before and
say so at startup; when it is on, an unauthenticated call never reaches a tool.
"""

from __future__ import annotations

from mcp.server.auth.middleware.auth_context import get_access_token
from mcp.server.auth.provider import AccessToken

from wald.db import SessionLocal
from wald.services import tokens


class WaldTokenVerifier:
    """Resolves a bearer token to the agent that owns it."""

    async def verify_token(self, token: str) -> AccessToken | None:
        with SessionLocal() as session:
            agent = tokens.resolve(session, token)
            if agent is None:
                return None
            # subject and client_id both carry the slug: subject is the identity the tools
            # read, client_id is what the SDK logs.
            return AccessToken(
                token=token,
                client_id=agent.slug,
                subject=agent.slug,
                scopes=["agent"],
                claims={"slug": agent.slug, "name": agent.name},
            )


def authenticated_slug() -> str | None:
    """The slug of the agent behind this request, or None when auth is off."""
    access = get_access_token()
    return access.subject if access else None


def caller_or(claimed: str | None, *, require: bool) -> str:
    """Resolve which agent a call is *from*.

    With auth on, the verified identity wins and the `from_agent` argument is ignored
    entirely -- accepting it would reintroduce the spoof it exists to prevent, and silently
    preferring the token while still reading the parameter is the kind of half-measure that
    reads as safe and is not.

    With auth off, the claimed value is all there is.
    """
    verified = authenticated_slug()
    if verified:
        return verified
    if require:
        raise PermissionError("authentication required: present a bearer token")
    if not claimed:
        raise ValueError("from_agent is required when authentication is disabled")
    return claimed
