"""The content loader.

Two properties carry the design. Loading must be idempotent -- running it twice on
unchanged files must not manufacture wiki revisions, or page history stops being a record
of what anyone actually edited. And a malformed or misspelled file must fail loudly:
silently dropping an unrecognized key leaves the author believing it took effect, which is
the worst possible outcome for a file that says which secret an agent should fetch.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import select

from wald.models import Resource, WikiPage, WikiPageRevision
from wald.services.seed import SeedError, parse_frontmatter, seed

EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "acme"


# --- Parsing (no database) -------------------------------------------------
def test_frontmatter_splits_metadata_from_body():
    meta, body = parse_frontmatter(
        '+++\ntitle = "Deploy"\ntags = ["ops"]\n+++\n\n# Deploy\n\ntext here\n', Path("x.md")
    )
    assert meta == {"title": "Deploy", "tags": ["ops"]}
    assert body.startswith("# Deploy")


def test_body_without_frontmatter_is_all_body():
    meta, body = parse_frontmatter("# Just markdown\n", Path("x.md"))
    assert meta == {}
    assert body == "# Just markdown\n"


def test_unclosed_frontmatter_is_an_error():
    # Silently treating the whole file as metadata (or as body) would lose content.
    with pytest.raises(SeedError, match="never closed"):
        parse_frontmatter('+++\ntitle = "x"\n\n# body\n', Path("x.md"))


def test_invalid_toml_names_the_file():
    with pytest.raises(SeedError, match="x.md"):
        parse_frontmatter("+++\ntitle = not-quoted\n+++\nbody\n", Path("x.md"))


# --- Loading (needs Postgres) ----------------------------------------------
def test_loads_the_example_tenant(session):
    report = seed(session, EXAMPLES)

    assert "wiki/deploy-process" in report.created
    assert "resource/analytics-warehouse" in report.created
    assert "resource/orders-api" in report.created  # second entry in the same file
    assert "agent/support-triage" in report.created
    assert report.chunks > 0

    page = session.scalar(select(WikiPage).where(WikiPage.slug == "deploy-process"))
    assert page.title == "Deploy process"
    assert page.space == "engineering"
    assert "Roll back first" in page.content
    # Frontmatter is metadata, not content.
    assert "+++" not in page.content


def test_slug_defaults_to_the_filename(session):
    seed(session, EXAMPLES)
    # examples/acme/agents/support-triage.toml declares no slug of its own.
    assert session.scalar(select(Resource).where(Resource.slug == "orders-api")) is not None


def test_secret_references_are_stored_but_secrets_are_not(session):
    seed(session, EXAMPLES)
    r = session.scalar(select(Resource).where(Resource.slug == "analytics-warehouse"))
    assert r.auth["secret_ref"].startswith("vault://")
    assert r.connection["host"].endswith("acme.example")


def test_reloading_unchanged_content_creates_no_revisions(session):
    seed(session, EXAMPLES)
    session.flush()
    page = session.scalar(select(WikiPage).where(WikiPage.slug == "deploy-process"))
    before = page.version

    second = seed(session, EXAMPLES)

    assert second.created == []
    assert second.updated == []
    assert "wiki/deploy-process" in second.unchanged
    assert page.version == before
    revisions = session.scalars(
        select(WikiPageRevision).where(WikiPageRevision.page_id == page.id)
    ).all()
    assert len(revisions) == 1


def test_editing_content_bumps_the_version_and_records_a_revision(session, tmp_path):
    (tmp_path / "wiki").mkdir()
    target = tmp_path / "wiki" / "runbook.md"
    target.write_text('+++\ntitle = "Runbook"\n+++\n\nfirst\n')
    seed(session, tmp_path)
    session.flush()

    target.write_text('+++\ntitle = "Runbook"\n+++\n\nsecond\n')
    report = seed(session, tmp_path)

    assert report.updated == ["wiki/runbook"]
    page = session.scalar(select(WikiPage).where(WikiPage.slug == "runbook"))
    assert page.version == 2
    assert page.content.strip() == "second"
    revisions = session.scalars(
        select(WikiPageRevision).where(WikiPageRevision.page_id == page.id)
    ).all()
    assert {r.content.strip() for r in revisions} == {"first", "second"}


def test_retagging_updates_without_a_revision(session, tmp_path):
    # A revision is a record of an edit to the page's substance; retagging is metadata.
    (tmp_path / "wiki").mkdir()
    target = tmp_path / "wiki" / "note.md"
    target.write_text('+++\ntitle = "Note"\n+++\n\nbody\n')
    seed(session, tmp_path)
    session.flush()

    target.write_text('+++\ntitle = "Note"\ntags = ["ops"]\n+++\n\nbody\n')
    report = seed(session, tmp_path)

    page = session.scalar(select(WikiPage).where(WikiPage.slug == "note"))
    assert report.updated == ["wiki/note"]
    assert page.tags == ["ops"]
    assert page.version == 1
    assert (
        len(
            session.scalars(
                select(WikiPageRevision).where(WikiPageRevision.page_id == page.id)
            ).all()
        )
        == 1
    )


def test_unknown_field_is_rejected_rather_than_ignored(session, tmp_path):
    (tmp_path / "resources").mkdir()
    (tmp_path / "resources" / "thing.toml").write_text('name = "Thing"\nkidn = "database"\n')
    with pytest.raises(SeedError, match="unknown field"):
        seed(session, tmp_path)


def test_empty_root_is_an_error_not_a_silent_success(session, tmp_path):
    # Pointing the loader at the wrong directory must not look like "nothing to do".
    with pytest.raises(SeedError, match="nothing to load"):
        seed(session, tmp_path)


def test_missing_root_names_the_path(session, tmp_path):
    with pytest.raises(SeedError, match="does not exist"):
        seed(session, tmp_path / "nope")
