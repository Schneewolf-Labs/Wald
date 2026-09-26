"""Wiki REST router."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from wald.db import get_session
from wald.models import WikiPage, WikiPageRevision
from wald.schemas import WikiPageIn, WikiPageOut, WikiPageUpdate
from wald.services import ingest

router = APIRouter(prefix="/wiki", tags=["wiki"])


def _get_page(session: Session, slug: str) -> WikiPage:
    page = session.scalar(select(WikiPage).where(WikiPage.slug == slug))
    if page is None:
        raise HTTPException(status_code=404, detail=f"wiki page '{slug}' not found")
    return page


@router.post("", response_model=WikiPageOut, status_code=201)
def create_page(body: WikiPageIn, session: Session = Depends(get_session)) -> WikiPage:
    if session.scalar(select(WikiPage).where(WikiPage.slug == body.slug)):
        raise HTTPException(status_code=409, detail=f"slug '{body.slug}' already exists")
    page = WikiPage(**body.model_dump())
    session.add(page)
    session.flush()
    session.add(
        WikiPageRevision(page_id=page.id, version=page.version, title=page.title, content=page.content)
    )
    ingest.index_wiki_page(session, page)
    session.commit()
    return page


@router.get("", response_model=list[WikiPageOut])
def list_pages(space: str | None = None, session: Session = Depends(get_session)) -> list[WikiPage]:
    stmt = select(WikiPage).order_by(WikiPage.updated_at.desc())
    if space:
        stmt = stmt.where(WikiPage.space == space)
    return list(session.scalars(stmt))


@router.get("/{slug}", response_model=WikiPageOut)
def get_page(slug: str, session: Session = Depends(get_session)) -> WikiPage:
    return _get_page(session, slug)


@router.patch("/{slug}", response_model=WikiPageOut)
def update_page(
    slug: str, body: WikiPageUpdate, session: Session = Depends(get_session)
) -> WikiPage:
    page = _get_page(session, slug)
    data = body.model_dump(exclude_unset=True)
    author = data.pop("author", None)
    for field, value in data.items():
        setattr(page, field, value)
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
    ingest.index_wiki_page(session, page)
    session.commit()
    return page
