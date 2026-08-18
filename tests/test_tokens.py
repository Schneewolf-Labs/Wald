"""Agent bearer tokens.

The property under test is the one that kept the hub on loopback: with a token, `from_agent`
stops being something a caller can assert. Everything else here exists to make sure that
guarantee cannot be quietly weakened -- a stored plaintext token, a token that outlives the
agent being retired, or a rotation that leaves the old one working.
"""

from __future__ import annotations

import pytest
from sqlalchemy import select

from wald.models import Agent
from wald.services import a2a, tokens


@pytest.fixture
def agent(session):
    a, _ = a2a.register(session, {"slug": "kira", "name": "Kira", "capabilities": ["coding"]})
    session.flush()
    return a


def test_issued_token_resolves_to_its_agent(session, agent):
    token = tokens.issue(session, agent)
    session.flush()
    assert tokens.resolve(session, token).id == agent.id


def test_the_token_itself_is_never_stored(session, agent):
    token = tokens.issue(session, agent)
    session.flush()
    stored = session.scalar(select(Agent.token_hash).where(Agent.id == agent.id))
    # A database dump should reveal who exists, not how to impersonate them.
    assert token not in (stored or "")
    assert stored == tokens.hash_token(token)
    assert len(stored) == 64


def test_a_wrong_token_resolves_to_nobody(session, agent):
    tokens.issue(session, agent)
    session.flush()
    assert tokens.resolve(session, "wald_not-a-real-token") is None
    assert tokens.resolve(session, "") is None


def test_reissuing_invalidates_the_previous_token(session, agent):
    first = tokens.issue(session, agent)
    session.flush()
    second = tokens.issue(session, agent)
    session.flush()
    # Rotation and revocation are the same operation, so two live tokens cannot coexist.
    assert tokens.resolve(session, first) is None
    assert tokens.resolve(session, second).id == agent.id


def test_revoking_stops_the_token_working(session, agent):
    token = tokens.issue(session, agent)
    session.flush()
    tokens.revoke(agent)
    session.flush()
    assert tokens.resolve(session, token) is None


def test_retiring_an_agent_disables_its_token(session, agent):
    # Otherwise retiring an agent silently leaves a working credential behind, and the person
    # retiring it has no reason to think a second step was needed.
    token = tokens.issue(session, agent)
    session.flush()
    agent.status = "retired"
    session.flush()
    assert tokens.resolve(session, token) is None


def test_tokens_are_unique_per_agent(session, agent):
    other, _ = a2a.register(session, {"slug": "scribe", "name": "Scribe"})
    session.flush()
    a = tokens.issue(session, agent)
    b = tokens.issue(session, other)
    session.flush()
    assert a != b
    assert tokens.resolve(session, a).slug == "kira"
    assert tokens.resolve(session, b).slug == "scribe"


def test_tokens_carry_real_entropy(session, agent):
    seen = {tokens.issue(session, agent) for _ in range(50)}
    assert len(seen) == 50
    sample = next(iter(seen))
    assert sample.startswith("wald_")
    assert len(sample) > 40  # 32 bytes urlsafe-encoded


# --- The identity guarantee ------------------------------------------------
def test_verified_identity_overrides_a_claimed_one():
    """With auth on, `from_agent` is ignored rather than merely deprioritised."""
    from unittest.mock import patch

    from wald.mcp import auth

    with patch.object(auth, "authenticated_slug", return_value="kira"):
        # The caller claims to be someone else; the token wins.
        assert auth.caller_or("scribe", require=True) == "kira"
        assert auth.caller_or(None, require=True) == "kira"


def test_no_token_is_refused_when_auth_is_required():
    from unittest.mock import patch

    from wald.mcp import auth

    with patch.object(auth, "authenticated_slug", return_value=None):
        with pytest.raises(PermissionError, match="authentication required"):
            auth.caller_or("kira", require=True)


def test_without_auth_the_claim_is_used_and_must_be_present():
    from unittest.mock import patch

    from wald.mcp import auth

    with patch.object(auth, "authenticated_slug", return_value=None):
        assert auth.caller_or("kira", require=False) == "kira"
        with pytest.raises(ValueError, match="from_agent is required"):
            auth.caller_or(None, require=False)
