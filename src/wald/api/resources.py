"""Resource directory REST router."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from wald.db import get_session
from wald.models import Resource
from wald.schemas import ResourceIn, ResourceOut
from wald.services import ingest

router = APIRouter(prefix="/resources", tags=["resources"])


@router.post("", response_model=ResourceOut, status_code=201)
def create_resource(body: ResourceIn, session: Session = Depends(get_session)) -> Resource:
    if session.scalar(select(Resource).where(Resource.slug == body.slug)):
        raise HTTPException(status_code=409, detail=f"slug '{body.slug}' already exists")
    resource = Resource(**body.model_dump())
    session.add(resource)
    session.flush()
    ingest.index_resource(session, resource)
    session.commit()
    return resource


@router.get("", response_model=list[ResourceOut])
def list_resources(
    kind: str | None = None, tag: str | None = None, session: Session = Depends(get_session)
) -> list[Resource]:
    stmt = select(Resource).order_by(Resource.name)
    if kind:
        stmt = stmt.where(Resource.kind == kind)
    if tag:
        stmt = stmt.where(Resource.tags.any(tag))
    return list(session.scalars(stmt))


@router.get("/{slug}", response_model=ResourceOut)
def get_resource(slug: str, session: Session = Depends(get_session)) -> Resource:
    resource = session.scalar(select(Resource).where(Resource.slug == slug))
    if resource is None:
        raise HTTPException(status_code=404, detail=f"resource '{slug}' not found")
    return resource
