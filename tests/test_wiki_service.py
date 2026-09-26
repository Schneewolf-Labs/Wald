"""The shared wiki upsert: versioning discipline under repeated writes."""

from __future__ import annotations

from sqlalchemy import select

from wald.models import WikiPageRevision
from wald.services import wiki as wiki_svc


def _revisions(session, page):
    return session.scalars(
        select(WikiPageRevision).where(WikiPageRevision.page_id == page.id)
    ).all()


def test_create_records_first_revision(session):
    page, created = wiki_svc.upsert_page(
        session, slug="notes", title="Notes", content="hello", author="kira"
    )
    assert created
    assert page.version == 1
    revs = _revisions(session, page)
    assert [r.version for r in revs] == [1]
    assert revs[0].author == "kira"


def test_identical_rewrite_is_not_a_version(session):
    wiki_svc.upsert_page(session, slug="notes", title="Notes", content="hello")
    page, created = wiki_svc.upsert_page(session, slug="notes", title="Notes", content="hello")
    assert not created
    assert page.version == 1
    assert len(_revisions(session, page)) == 1


def test_content_change_bumps_version(session):
    wiki_svc.upsert_page(session, slug="notes", title="Notes", content="hello")
    page, _ = wiki_svc.upsert_page(session, slug="notes", title="Notes", content="goodbye")
    assert page.version == 2
    assert [r.version for r in _revisions(session, page)] == [1, 2]


def test_omitted_space_and_tags_are_kept_on_update(session):
    wiki_svc.upsert_page(
        session, slug="notes", title="Notes", content="hello", space="agent-notes", tags=["a"]
    )
    page, _ = wiki_svc.upsert_page(session, slug="notes", title="Notes", content="changed")
    assert page.space == "agent-notes"
    assert page.tags == ["a"]
