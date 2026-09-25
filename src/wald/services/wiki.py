"""Wiki writes as a service, shared by whichever surface accepts them.

The REST router predates this module and still carries its own create/update handlers;
the MCP write tool starts here. The upsert follows the seed loader's discipline: a
revision records an edit to the page's *substance*, so writing identical content back is
a no-op rather than a manufactured version bump -- an agent that re-asserts what a page
already says must not bury the history of what anyone actually changed.
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from wald.models import WikiPage, WikiPageRevision


def upsert_page(
    session: Session,
    *,
    slug: str,
    title: str,
    content: str,
    space: str | None = None,
    tags: list[str] | None = None,
    author: str | None = None,
) -> tuple[WikiPage, bool]:
    """Create or update a page. Returns ``(page, created)``.

    ``space`` and ``tags`` left as None mean "keep what the page has" on update, and the
    usual defaults on create. The caller is responsible for authorization and for
    committing (via services/background.finish_write, so indexing follows its policy).
    """
    page = session.scalar(select(WikiPage).where(WikiPage.slug == slug))
    if page is None:
        page = WikiPage(
            slug=slug,
            title=title,
            content=content,
            space=space or "general",
            tags=list(tags or []),
        )
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
        return page, True

    if space is not None:
        page.space = space
    if tags is not None:
        page.tags = list(tags)
    if (title, content) != (page.title, page.content):
        page.title = title
        page.content = content
        page.version += 1
        session.add(
            WikiPageRevision(
                page_id=page.id,
                version=page.version,
                title=title,
                content=content,
                author=author,
            )
        )
    return page, False
