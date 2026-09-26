"""Authorization: grant parsing, matching, and what enforcement changes."""

from __future__ import annotations

import pytest

from wald.services import authz


# --- Grammar --------------------------------------------------------------
def test_parse_grant_full_form():
    g = authz.parse_grant("wiki:read:engineering")
    assert (g.pillar, g.action, g.selector) == ("wiki", "read", "engineering")


def test_parse_grant_selector_defaults_to_wildcard():
    assert authz.parse_grant("resource:read").selector == authz.ALL


@pytest.mark.parametrize(
    "bad",
    [
        "wiki",  # no action
        "wiki:read:a:b",  # too many segments
        "wiki::x",  # empty action
        "wiki:destroy",  # unknown action
        "mailbox:read",  # unknown pillar
        "resource:delete:x",  # known pillar, unknown action
    ],
)
def test_parse_grant_rejects_outside_grammar(bad):
    with pytest.raises(ValueError):
        authz.parse_grant(bad)


# --- Matching -------------------------------------------------------------
def test_wildcard_selector_matches_everything():
    grants = authz.Grants(["wiki:read:*"])
    assert grants.allows("wiki", "read", "engineering")
    assert grants.allows("wiki", "read", "anything-else")
    assert not grants.allows("wiki", "write", "engineering")


def test_concrete_selector_matches_only_itself():
    grants = authz.Grants(["resource:read:merlina"])
    assert grants.allows("resource", "read", "merlina")
    assert not grants.allows("resource", "read", "spark")


def test_empty_grants_deny_everything():
    grants = authz.Grants([])
    assert not grants.allows("wiki", "read", "general")


def test_allow_all_is_unrestricted():
    assert authz.ALLOW_ALL.unrestricted
    assert authz.ALLOW_ALL.allows("wiki", "write", "anything")


def test_grants_reject_malformed_strings_at_construction():
    with pytest.raises(ValueError):
        authz.Grants(["wiki:read:*", "nonsense"])


# --- Search scoping (needs Postgres) --------------------------------------
def _make_page(session, slug, space, text):
    from wald.models import WikiPage
    from wald.services import ingest

    page = WikiPage(slug=slug, title=slug, space=space, content=text)
    session.add(page)
    session.flush()
    ingest.index_wiki_page(session, page)
    return page


def test_search_scopes_to_readable_spaces(session):
    from wald.services import search as search_svc

    _make_page(session, "public-runbook", "general", "the shared deployment runbook")
    _make_page(session, "secret-runbook", "restricted", "the restricted deployment runbook")
    session.flush()

    unrestricted = search_svc.search(session, "deployment runbook", top_k=5)
    assert {h.title for h in unrestricted} >= {"public-runbook", "secret-runbook"}

    scoped = search_svc.search(
        session, "deployment runbook", top_k=5, grants=authz.Grants(["wiki:read:general"])
    )
    titles = {h.title for h in scoped}
    assert "public-runbook" in titles
    assert "secret-runbook" not in titles


def test_search_scopes_resources_by_slug(session):
    from wald.models import Resource
    from wald.services import ingest
    from wald.services import search as search_svc

    for slug in ("alpha-db", "beta-db"):
        r = Resource(slug=slug, name=slug, kind="database", description=f"{slug} customer database")
        session.add(r)
        session.flush()
        ingest.index_resource(session, r)
    session.flush()

    scoped = search_svc.search(
        session, "customer database", top_k=5, grants=authz.Grants(["resource:read:alpha-db"])
    )
    titles = {h.title for h in scoped}
    assert "alpha-db" in titles
    assert "beta-db" not in titles
