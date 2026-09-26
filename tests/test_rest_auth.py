"""The REST surface enforces the same identity and grants as MCP.

Before this, authentication and authorization stopped at the MCP server, so the REST port
was a way around both: send as any agent, read any inbox, read what grants withheld.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from wald.config import get_settings
from wald.db import get_session
from wald.main import create_app
from wald.models import Agent, Resource, WikiPage, WikiPageRevision
from wald.services import a2a, ingest, tokens


def _client(session, **settings):
    app = create_app()
    app.dependency_overrides[get_session] = lambda: session
    app.dependency_overrides[get_settings] = lambda: get_settings().model_copy(update=settings)
    return TestClient(app)


@pytest.fixture
def hub(session):
    alice = Agent(
        slug="alice",
        name="Alice",
        grants=["wiki:read:engineering", "wiki:write:engineering", "resource:read:merlina"],
    )
    bob = Agent(slug="bob", name="Bob")
    carol = Agent(slug="carol", name="Carol")
    session.add_all([alice, bob, carol])
    for slug, space in (("runbook", "engineering"), ("salaries", "hr")):
        page = WikiPage(slug=slug, title=slug.title(), space=space, content=f"{slug} vpn notes")
        session.add(page)
        session.flush()
        ingest.index_wiki_page(session, page)
    session.add_all(
        [
            Resource(slug="merlina", name="Merlina", kind="service"),
            Resource(slug="vault", name="Vault", kind="service"),
        ]
    )
    session.flush()
    return {
        "alice": {"Authorization": f"Bearer {tokens.issue(session, alice)}"},
        "bob": {"Authorization": f"Bearer {tokens.issue(session, bob)}"},
        "carol": {"Authorization": f"Bearer {tokens.issue(session, carol)}"},
    }


@pytest.fixture
def authn(session, hub):
    return _client(session, require_auth=True)


@pytest.fixture
def authz(session, hub):
    return _client(session, require_auth=True, enforce_authz=True)


# --- Authentication ---------------------------------------------------------
def test_without_auth_the_api_behaves_as_before(session, hub):
    client = _client(session)
    assert len(client.get("/wiki").json()) == 2
    r = client.post(
        "/agents/messages", json={"from_agent": "bob", "to_agent": "alice", "content": "hi"}
    )
    assert r.status_code == 201


@pytest.mark.parametrize("path", ["/wiki", "/resources", "/agents", "/search?q=vpn"])
def test_auth_on_turns_away_a_request_without_a_token(authn, path):
    r = authn.get(path)
    assert r.status_code == 401
    assert r.headers["www-authenticate"] == "Bearer"


def test_health_stays_open(authn):
    assert authn.get("/health").status_code == 200


def test_a_bad_token_is_rejected_even_with_auth_off(session, hub):
    # Quietly treating it as anonymous would turn a typo into a request made as nobody.
    client = _client(session)
    assert client.get("/wiki", headers={"Authorization": "Bearer wald_nope"}).status_code == 401


def test_a_revoked_token_stops_working(session, authn, hub):
    tokens.revoke(a2a.resolve_agent(session, "bob"))
    session.flush()
    assert authn.get("/wiki", headers=hub["bob"]).status_code == 401


# --- A2A --------------------------------------------------------------------
def test_the_sender_is_the_token_not_the_claim(authn, hub):
    r = authn.post(
        "/agents/messages",
        headers=hub["bob"],
        json={"from_agent": "alice", "to_agent": "carol", "content": "trust me"},
    )
    assert r.status_code == 201
    bob = authn.get("/agents/bob", headers=hub["bob"]).json()
    inbox = authn.get("/agents/carol/inbox", headers=hub["carol"]).json()
    assert [m["from_agent_id"] for m in inbox] == [bob["id"]]


def test_an_agent_reads_only_its_own_inbox(authn, hub):
    assert authn.get("/agents/alice/inbox", headers=hub["bob"]).status_code == 403
    assert authn.get("/agents/bob/inbox", headers=hub["bob"]).status_code == 200


def test_a_thread_belongs_to_its_participants(authn, hub):
    sent = authn.post(
        "/agents/messages", headers=hub["alice"], json={"to_agent": "bob", "content": "hi"}
    ).json()
    thread = sent["thread_id"]
    assert authn.get(f"/agents/threads/{thread}", headers=hub["bob"]).status_code == 200
    assert authn.get(f"/agents/threads/{thread}", headers=hub["carol"]).status_code == 404
    # Nor can an outsider post into it.
    r = authn.post(
        "/agents/messages",
        headers=hub["carol"],
        json={"to_agent": "bob", "content": "me too", "thread_id": thread},
    )
    assert r.status_code == 404


def test_an_agent_cannot_register_over_another(authn, hub):
    r = authn.post(
        "/agents",
        headers=hub["bob"],
        json={"slug": "alice", "name": "Alice", "endpoint_url": "http://evil"},
    )
    assert r.status_code == 403


# --- Authorization ----------------------------------------------------------
def test_wiki_reads_follow_grants(authz, hub):
    assert [p["slug"] for p in authz.get("/wiki", headers=hub["alice"]).json()] == ["runbook"]
    assert authz.get("/wiki/runbook", headers=hub["alice"]).status_code == 200
    # Unreadable reads as absent, so the API is no oracle for what exists.
    assert authz.get("/wiki/salaries", headers=hub["alice"]).status_code == 404
    assert authz.get("/wiki", headers=hub["bob"]).json() == []


def test_wiki_writes_follow_grants(authz, hub):
    page = {"slug": "new", "title": "New", "content": "x"}
    r = authz.post("/wiki", headers=hub["alice"], json={**page, "space": "hr"})
    assert r.status_code == 403
    r = authz.post("/wiki", headers=hub["alice"], json={**page, "space": "engineering"})
    assert r.status_code == 201
    assert (
        authz.patch("/wiki/salaries", headers=hub["alice"], json={"content": "y"}).status_code
        == 404
    )


def test_the_author_is_the_token_not_the_claim(session, authz, hub):
    r = authz.patch(
        "/wiki/runbook", headers=hub["alice"], json={"content": "updated", "author": "bob"}
    )
    assert r.status_code == 200
    page = session.query(WikiPage).filter_by(slug="runbook").one()
    latest = (
        session.query(WikiPageRevision)
        .filter_by(page_id=page.id)
        .order_by(WikiPageRevision.version.desc())
        .first()
    )
    assert latest.author == "alice"


def test_resources_follow_grants(authz, hub):
    assert [r["slug"] for r in authz.get("/resources", headers=hub["alice"]).json()] == ["merlina"]
    assert authz.get("/resources/vault", headers=hub["alice"]).status_code == 404
    r = authz.post(
        "/resources", headers=hub["alice"], json={"slug": "new", "name": "New", "kind": "tool"}
    )
    assert r.status_code == 403


def test_search_is_scoped_to_what_the_caller_may_read(authz, hub):
    hits = authz.get("/search?q=vpn", headers=hub["alice"]).json()["hits"]
    assert hits
    assert {h["title"] for h in hits if h["source_type"] == "wiki"} == {"Runbook"}


def test_ask_is_scoped_to_what_the_caller_may_read(authz, hub):
    body = authz.post("/ask", headers=hub["alice"], json={"question": "vpn"}).json()
    assert body["citations"]
    assert "Salaries" not in {c["title"] for c in body["citations"]}


# --- Web UI -----------------------------------------------------------------
def test_the_web_ui_closes_with_auth_on(session, authn):
    r = authn.get("/")
    assert r.status_code == 403
    assert "WALD_WEB_UI_OPEN" in r.text
    assert authn.get("/ui/wiki/salaries").status_code == 403


def test_the_web_ui_opens_when_the_operator_says_so(session, hub):
    client = _client(session, require_auth=True, web_ui_open=True)
    assert client.get("/").status_code == 200


# --- MCP threads ------------------------------------------------------------
def test_mcp_threads_belong_to_their_participants(session, hub, monkeypatch):
    """The MCP tools had the same gap: any verified agent could read or post into any thread."""
    from unittest.mock import patch

    from mcp.server.fastmcp.exceptions import ToolError
    from sqlalchemy.orm import Session

    from wald.mcp import auth as mcp_auth
    from wald.mcp import server

    # The tools open their own sessions; bind them to the test transaction.
    monkeypatch.setattr(
        server,
        "SessionLocal",
        lambda: Session(bind=session.connection(), join_transaction_mode="create_savepoint"),
    )
    alice, bob = a2a.resolve_agent(session, "alice"), a2a.resolve_agent(session, "bob")
    thread = str(a2a.send_message(session, from_agent=alice, to_agent=bob, content="hi").thread_id)
    session.flush()

    def as_agent(slug):
        # caller_or reads the identity through wald.mcp.auth, the thread checks through
        # the server module's import of the same function.
        monkeypatch.setattr(mcp_auth, "authenticated_slug", lambda: slug)
        return patch.object(server, "authenticated_slug", return_value=slug)

    with as_agent("bob"):
        assert len(server.get_conversation(thread)) == 1
    with as_agent("carol"):
        with pytest.raises(ToolError, match="no thread"):
            server.get_conversation(thread)
        with pytest.raises(ToolError, match="no thread"):
            server.send_agent_message(to_agent="bob", content="me too", thread_id=thread)
