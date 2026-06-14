"""Agent directory + A2A REST router."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from wald.db import get_session
from wald.models import Agent
from wald.schemas import AgentIn, AgentMessageIn, AgentMessageOut, AgentOut
from wald.services import a2a, ingest

router = APIRouter(prefix="/agents", tags=["agents"])


@router.post("", response_model=AgentOut, status_code=201)
def register_agent(body: AgentIn, session: Session = Depends(get_session)) -> Agent:
    if session.scalar(select(Agent).where(Agent.slug == body.slug)):
        raise HTTPException(status_code=409, detail=f"slug '{body.slug}' already exists")
    agent = Agent(**body.model_dump())
    session.add(agent)
    session.flush()
    ingest.index_agent(session, agent)
    session.commit()
    return agent


@router.get("", response_model=list[AgentOut])
def list_agents(
    capability: str | None = None, session: Session = Depends(get_session)
) -> list[Agent]:
    if capability:
        return a2a.find_agents_by_capability(session, capability)
    return list(session.scalars(select(Agent).order_by(Agent.name)))


@router.get("/{slug}", response_model=AgentOut)
def get_agent(slug: str, session: Session = Depends(get_session)) -> Agent:
    agent = a2a.resolve_agent(session, slug)
    if agent is None:
        raise HTTPException(status_code=404, detail=f"agent '{slug}' not found")
    return agent


@router.post("/messages", response_model=AgentMessageOut, status_code=201)
def send_message(body: AgentMessageIn, session: Session = Depends(get_session)) -> AgentMessageOut:
    sender = a2a.resolve_agent(session, body.from_agent)
    target = a2a.resolve_agent(session, body.to_agent)
    if sender is None or target is None:
        raise HTTPException(status_code=404, detail="sender or target agent not found")
    msg = a2a.send_message(
        session,
        from_agent=sender,
        to_agent=target,
        content=body.content,
        role=body.role,
        thread_id=body.thread_id,
    )
    session.commit()
    return AgentMessageOut.model_validate(msg)


@router.get("/{slug}/inbox", response_model=list[AgentMessageOut])
def get_inbox(
    slug: str, unread_only: bool = False, session: Session = Depends(get_session)
) -> list[AgentMessageOut]:
    agent = a2a.resolve_agent(session, slug)
    if agent is None:
        raise HTTPException(status_code=404, detail=f"agent '{slug}' not found")
    return [AgentMessageOut.model_validate(m) for m in a2a.inbox(session, agent, unread_only=unread_only)]


@router.get("/threads/{thread_id}", response_model=list[AgentMessageOut])
def get_thread(thread_id: uuid.UUID, session: Session = Depends(get_session)) -> list[AgentMessageOut]:
    return [AgentMessageOut.model_validate(m) for m in a2a.thread(session, thread_id)]
