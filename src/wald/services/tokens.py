"""Agent bearer tokens: issuing, verifying, revoking.

This exists to close the gap that has been the top of the roadmap since the MCP surface went
up. `send_agent_message` took the sender's identity as an *argument*, so any caller could
claim to be any agent. That is survivable on a trusted network and not survivable anywhere
else, and it is what kept the hub bound to loopback.

With a token, identity becomes something the caller proves rather than something it asserts,
and the tools stop trusting a parameter.

Three choices worth stating:

**Only the hash is stored.** The hub needs the means to *check* an identity, not to present
one. A database dump then reveals which agents exist, not how to impersonate them, and there
is no recovery path for a lost token other than issuing a new one -- which is correct.

**SHA-256, not a password hash.** Argon2 and bcrypt exist to make brute force expensive
against *low-entropy* human-chosen secrets. These tokens carry 256 bits from
`secrets.token_urlsafe(32)`, so there is nothing to brute force, and a slow hash on the
verification path would only cost latency on every single tool call.

**Comparison is constant-time.** The lookup is by hash, which is already not attacker-
controlled in a useful way, but the final equality check uses `compare_digest` so verification
time carries no signal about how much of a candidate hash matched.
"""

from __future__ import annotations

import hashlib
import secrets

from sqlalchemy import select
from sqlalchemy.orm import Session

from wald.models import Agent

TOKEN_BYTES = 32  # 256 bits
PREFIX = "wald_"


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def issue(session: Session, agent: Agent) -> str:
    """Mint a new token for an agent and store only its hash.

    Returns the token, which is the one and only time it exists in readable form. Issuing
    again invalidates the previous token -- rotation and revocation are the same operation,
    so there is no way to end up with two live tokens for one agent by accident.
    """
    token = PREFIX + secrets.token_urlsafe(TOKEN_BYTES)
    agent.token_hash = hash_token(token)
    return token


def revoke(agent: Agent) -> None:
    agent.token_hash = None


def resolve(session: Session, token: str) -> Agent | None:
    """Return the agent a token belongs to, or None.

    Rejects agents that are not active: retiring an agent should stop its token working
    without anyone having to remember to revoke it separately.
    """
    if not token:
        return None
    candidate = hash_token(token)
    agent = session.scalar(
        select(Agent).where(Agent.token_hash == candidate, Agent.status == "active")
    )
    if agent is None:
        return None
    # Constant-time even though the lookup already matched: keeps verification timing free of
    # any signal about how close a wrong token was.
    if not secrets.compare_digest(agent.token_hash or "", candidate):
        return None
    return agent


def token_cli() -> None:
    """Console-script entry point (``wald-token``)."""
    import argparse

    from wald.db import SessionLocal
    from wald.services import a2a

    parser = argparse.ArgumentParser(description="Issue or revoke an agent's bearer token.")
    parser.add_argument("agent", help="agent slug")
    parser.add_argument("--revoke", action="store_true", help="revoke instead of issuing")
    args = parser.parse_args()

    with SessionLocal() as session:
        agent = a2a.resolve_agent(session, args.agent)
        if agent is None:
            raise SystemExit(f"wald-token: no agent '{args.agent}'")
        if args.revoke:
            revoke(agent)
            session.commit()
            print(f"revoked token for {agent.slug}")
            return
        token = issue(session, agent)
        session.commit()
        # Printed once and never recoverable: only the hash is stored. Saying so here is the
        # difference between someone copying it now and someone filing a bug later.
        print(f"agent : {agent.slug}")
        print(f"token : {token}")
        print("\nStore it now -- only its hash is kept, so this cannot be shown again.")
        print("Issuing again replaces it; --revoke disables access without issuing a new one.")
