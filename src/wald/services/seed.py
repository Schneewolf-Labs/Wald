"""Load file-backed content into the hub.

The organizing idea: a directory of plain files is the source of truth, and the database
is a derived index of it. That buys three things a web form does not. Content lives
wherever the organization already keeps its private material -- a separate repo, an
encrypted volume -- instead of inside a public codebase. Review and history come from
whatever version control already wraps that directory. And rebuilding the hub from
scratch is one command, which is what makes the database safe to drop.

Layout::

    <root>/wiki/*.md          markdown with TOML frontmatter (+++ fenced)
    <root>/resources/*.toml    resource directory entries
    <root>/agents/*.toml       agent registry entries

Slugs default to the filename stem, so `wiki/onboarding.md` is `onboarding` and renaming
a file is a deliberate act rather than an accident.

Loading is an idempotent upsert keyed by slug, and unchanged entities are skipped
outright. That last part is not an optimization: re-running the loader on a wiki page
that has not changed must not manufacture a revision, or the page history fills with
noise and stops being a record of what anyone actually edited.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from wald.models import Agent, Resource, WikiPage, WikiPageRevision
from wald.services import ingest

_FENCE = "+++"

# Fields the loader is allowed to set, per entity. Anything else in a file is a typo or a
# stale key, and silently dropping it would leave the author believing it took effect.
_RESOURCE_FIELDS = {
    "slug",
    "name",
    "kind",
    "description",
    "connection",
    "auth",
    "docs_url",
    "owner",
    "tags",
    "status",
}
_AGENT_FIELDS = {
    "slug",
    "name",
    "description",
    "capabilities",
    "endpoint_url",
    "protocol",
    "auth",
    "owner",
    "status",
    "agent_card",
}
_WIKI_FIELDS = {"slug", "title", "space", "tags", "author"}


class SeedError(Exception):
    """A content file could not be loaded. Carries the path so the fix is obvious."""


@dataclass
class SeedReport:
    created: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    chunks: int = 0

    def summary(self) -> str:
        return (
            f"{len(self.created)} created, {len(self.updated)} updated, "
            f"{len(self.unchanged)} unchanged, {self.chunks} chunks embedded"
        )


def parse_frontmatter(text: str, source: Path) -> tuple[dict[str, Any], str]:
    """Split a `+++`-fenced TOML frontmatter block from its markdown body.

    TOML rather than YAML because Python 3.12 parses TOML in the standard library, and a
    content loader should not drag in a parser dependency to read its own metadata.
    """
    if not text.startswith(_FENCE):
        return {}, text

    end = text.find(f"\n{_FENCE}", len(_FENCE))
    if end == -1:
        raise SeedError(f"{source}: frontmatter opened with '{_FENCE}' but never closed")

    raw = text[len(_FENCE) : end]
    body = text[end + len(_FENCE) + 1 :].lstrip("\n")
    try:
        meta = tomllib.loads(raw)
    except tomllib.TOMLDecodeError as exc:
        raise SeedError(f"{source}: invalid TOML frontmatter: {exc}") from exc
    return meta, body


def _load_toml(path: Path) -> dict[str, Any]:
    try:
        return tomllib.loads(path.read_text())
    except tomllib.TOMLDecodeError as exc:
        raise SeedError(f"{path}: invalid TOML: {exc}") from exc


def _entries(path: Path, plural: str) -> list[dict[str, Any]]:
    """One file may hold a single entity, or several under an array-of-tables.

    Grouping related entries (every GPU host, say) in one file is often the natural way to
    write them, so `[[resource]]` is supported alongside the one-per-file form.
    """
    doc = _load_toml(path)
    if plural in doc:
        rows = doc[plural]
        if not isinstance(rows, list):
            raise SeedError(f"{path}: '{plural}' must be an array of tables")
        return rows
    return [doc]


def _check_fields(data: dict[str, Any], allowed: set[str], path: Path) -> None:
    unknown = set(data) - allowed
    if unknown:
        raise SeedError(f"{path}: unknown field(s): {', '.join(sorted(unknown))}")


def _changed(obj: Any, data: dict[str, Any]) -> dict[str, Any]:
    """Fields whose desired value differs from what is stored."""
    return {k: v for k, v in data.items() if getattr(obj, k, None) != v}


def _seed_wiki(session: Session, root: Path, report: SeedReport) -> None:
    for path in sorted(root.glob("*.md")):
        meta, body = parse_frontmatter(path.read_text(), path)
        _check_fields(meta, _WIKI_FIELDS, path)

        slug = meta.get("slug", path.stem)
        author = meta.get("author", "wald-seed")
        desired = {
            "slug": slug,
            "title": meta.get("title", slug.replace("-", " ").title()),
            "space": meta.get("space", "general"),
            "content": body,
            "tags": list(meta.get("tags", [])),
        }

        page = session.scalar(select(WikiPage).where(WikiPage.slug == slug))
        if page is None:
            page = WikiPage(**desired)
            session.add(page)
            session.flush()
            session.add(
                WikiPageRevision(
                    page_id=page.id,
                    version=page.version,
                    title=page.title,
                    content=page.content,
                    author=author,
                )
            )
            report.chunks += ingest.index_wiki_page(session, page)
            report.created.append(f"wiki/{slug}")
            continue

        diff = _changed(page, desired)
        if not diff:
            report.unchanged.append(f"wiki/{slug}")
            continue

        for key, value in diff.items():
            setattr(page, key, value)
        # A revision records an edit to the page's substance. Retagging is metadata and
        # does not deserve an entry in the history a reader is trying to follow.
        if {"title", "content"} & set(diff):
            page.version += 1
            session.add(
                WikiPageRevision(
                    page_id=page.id,
                    version=page.version,
                    title=page.title,
                    content=page.content,
                    author=author,
                )
            )
        report.chunks += ingest.index_wiki_page(session, page)
        report.updated.append(f"wiki/{slug}")


def _seed_simple(
    session: Session,
    root: Path,
    *,
    model: type,
    plural: str,
    allowed: set[str],
    indexer: Any,
    label: str,
    report: SeedReport,
) -> None:
    for path in sorted(root.glob("*.toml")):
        for i, data in enumerate(_entries(path, plural)):
            _check_fields(data, allowed, path)
            slug = data.get("slug") or (path.stem if i == 0 else None)
            if not slug:
                raise SeedError(f"{path}: entry {i} needs an explicit 'slug'")
            data = {**data, "slug": slug}
            if "name" not in data:
                data["name"] = slug.replace("-", " ").title()

            obj = session.scalar(select(model).where(model.slug == slug))
            if obj is None:
                obj = model(**data)
                session.add(obj)
                session.flush()
                report.chunks += indexer(session, obj)
                report.created.append(f"{label}/{slug}")
                continue

            diff = _changed(obj, data)
            if not diff:
                report.unchanged.append(f"{label}/{slug}")
                continue
            for key, value in diff.items():
                setattr(obj, key, value)
            report.chunks += indexer(session, obj)
            report.updated.append(f"{label}/{slug}")


def seed(session: Session, root: Path) -> SeedReport:
    """Load every content file under `root`. Idempotent."""
    root = root.expanduser()
    if not root.is_dir():
        raise SeedError(f"content root does not exist: {root}")

    report = SeedReport()
    if not any((root / d).is_dir() for d in ("wiki", "resources", "agents")):
        raise SeedError(
            f"{root} has no wiki/, resources/ or agents/ subdirectory -- nothing to load"
        )
    if (root / "wiki").is_dir():
        _seed_wiki(session, root / "wiki", report)
    if (root / "resources").is_dir():
        _seed_simple(
            session,
            root / "resources",
            model=Resource,
            plural="resource",
            allowed=_RESOURCE_FIELDS,
            indexer=ingest.index_resource,
            label="resource",
            report=report,
        )
    if (root / "agents").is_dir():
        _seed_simple(
            session,
            root / "agents",
            model=Agent,
            plural="agent",
            allowed=_AGENT_FIELDS,
            indexer=ingest.index_agent,
            label="agent",
            report=report,
        )
    return report


def seed_cli() -> None:
    """Console-script entry point (``wald-seed``)."""
    import argparse

    from wald.config import get_settings
    from wald.db import SessionLocal

    parser = argparse.ArgumentParser(description="Load file-backed content into Wald.")
    parser.add_argument(
        "root",
        nargs="?",
        default=None,
        help="content directory (default: $WALD_CONTENT_DIR, else ./content)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="report what would change, then roll back",
    )
    args = parser.parse_args()
    root = Path(args.root or get_settings().content_dir)

    with SessionLocal() as session:
        try:
            report = seed(session, root)
        except SeedError as exc:
            raise SystemExit(f"wald-seed: {exc}") from exc
        # A dry run exercises the real path and discards it, rather than predicting from a
        # parallel code path that could disagree with the one that actually writes.
        if args.dry_run:
            session.rollback()
        else:
            session.commit()

    for slug in report.created:
        print(f"  + {slug}")
    for slug in report.updated:
        print(f"  ~ {slug}")
    prefix = "would apply" if args.dry_run else "applied"
    print(f"wald-seed: {prefix} {report.summary()} from {root}")
