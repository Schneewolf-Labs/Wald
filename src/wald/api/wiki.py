"""Wiki REST router."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from wald.api.auth import Caller, get_caller, require
from wald.db import get_session
from wald.models import WikiPage, WikiPageRevision
from wald.schemas import WikiPageIn, WikiPageOut, WikiPageUpdate
from wald.services import background

router = APIRouter(prefix="/wiki", tags=["wiki"])


def _get_page(session: Session, slug: str, caller: Caller) -> WikiPage:
    page = session.scalar(select(WikiPage).where(WikiPage.slug == slug))
    # A page the caller may not read is reported as absent, the same answer the listing
    # gives, so the API is not an oracle for which slugs exist in which spaces.
    if page is None or not caller.grants.allows("wiki", "read", page.space):
        raise HTTPException(status_code=404, detail=f"wiki page '{slug}' not found")
    return page


@router.post("", response_model=WikiPageOut, status_code=201)
def create_page(
    body: WikiPageIn,
    session: Session = Depends(get_session),
    caller: Caller = Depends(get_caller),
) -> WikiPage:
    require(caller, "wiki", "write", body.space)
    if session.scalar(select(WikiPage).where(WikiPage.slug == body.slug)):
        raise HTTPException(status_code=409, detail=f"slug '{body.slug}' already exists")
    page = WikiPage(**body.model_dump())
    session.add(page)
    session.flush()
    session.add(
        WikiPageRevision(
            page_id=page.id,
            version=page.version,
            title=page.title,
            content=page.content,
            author=caller.slug,
        )
    )
    background.finish_write(session, "wiki", page)
    return page


@router.get("", response_model=list[WikiPageOut])
def list_pages(
    space: str | None = None,
    session: Session = Depends(get_session),
    caller: Caller = Depends(get_caller),
) -> list[WikiPage]:
    stmt = select(WikiPage).order_by(WikiPage.updated_at.desc())
    if space:
        stmt = stmt.where(WikiPage.space == space)
    return [p for p in session.scalars(stmt) if caller.grants.allows("wiki", "read", p.space)]


@router.get("/{slug}", response_model=WikiPageOut)
def get_page(
    slug: str, session: Session = Depends(get_session), caller: Caller = Depends(get_caller)
) -> WikiPage:
    return _get_page(session, slug, caller)


@router.patch("/{slug}", response_model=WikiPageOut)
def update_page(
    slug: str,
    body: WikiPageUpdate,
    session: Session = Depends(get_session),
    caller: Caller = Depends(get_caller),
) -> WikiPage:
    page = _get_page(session, slug, caller)
    require(caller, "wiki", "write", page.space)
    data = body.model_dump(exclude_unset=True)
    # The verified identity is the author when there is one; the field is only honoured
    # without authentication, where it is all there is.
    author = caller.slug or data.pop("author", None)
    data.pop("author", None)
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
    background.finish_write(session, "wiki", page)
    return page
