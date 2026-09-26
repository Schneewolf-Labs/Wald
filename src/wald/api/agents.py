"""Agent directory + A2A REST router.

With authentication on, this follows the MCP tools exactly: the sender of a message is
the token's agent and `from_agent` is ignored, an inbox is readable only by its owner, a
thread only by its participants, and an agent may only register itself.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from wald.api.auth import Caller, get_caller
from wald.db import get_session
from wald.models import Agent
from wald.schemas import AgentIn, AgentMessageIn, AgentMessageOut, AgentOut
from wald.services import a2a, background

router = APIRouter(prefix="/agents", tags=["agents"])


@router.post("", response_model=AgentOut, status_code=201)
def register_agent(
    body: AgentIn,
    session: Session = Depends(get_session),
    caller: Caller = Depends(get_caller),
) -> Agent:
    # Peers send their credentials to a registered endpoint_url, so registering under
    # another agent's slug would redirect its traffic. (An authenticated agent already has
    # a row -- its token lives there -- so for it this ends in the 409 below; new agents
    # arrive through the seed loader.)
    if caller.slug and caller.slug != body.slug:
        raise HTTPException(
            status_code=403,
            detail=f"authenticated as '{caller.slug}'; an agent may only register itself",
        )
    if session.scalar(select(Agent).where(Agent.slug == body.slug)):
        raise HTTPException(status_code=409, detail=f"slug '{body.slug}' already exists")
    agent = Agent(**body.model_dump())
    session.add(agent)
    session.flush()
    background.finish_write(session, "agent", agent)
    return agent


@router.get("", response_model=list[AgentOut])
def list_agents(
    capability: str | None = None,
    session: Session = Depends(get_session),
    caller: Caller = Depends(get_caller),
) -> list[Agent]:
    if capability:
        return a2a.find_agents_by_capability(session, capability)
    return list(session.scalars(select(Agent).order_by(Agent.name)))


@router.get("/{slug}", response_model=AgentOut)
def get_agent(
    slug: str, session: Session = Depends(get_session), caller: Caller = Depends(get_caller)
) -> Agent:
    agent = a2a.resolve_agent(session, slug)
    if agent is None:
        raise HTTPException(status_code=404, detail=f"agent '{slug}' not found")
    return agent


@router.post("/messages", response_model=AgentMessageOut, status_code=201)
def send_message(
    body: AgentMessageIn,
    session: Session = Depends(get_session),
    caller: Caller = Depends(get_caller),
) -> AgentMessageOut:
    if caller.agent is not None:
        sender = caller.agent
    elif body.from_agent:
        sender = a2a.resolve_agent(session, body.from_agent)
    else:
        raise HTTPException(
            status_code=422, detail="from_agent is required when authentication is disabled"
        )
    target = a2a.resolve_agent(session, body.to_agent)
    if sender is None or target is None:
        raise HTTPException(status_code=404, detail="sender or target agent not found")
    if (
        caller.agent is not None
        and body.thread_id is not None
        and not a2a.in_thread(session, body.thread_id, caller.agent)
    ):
        # Continuing a thread is for its participants; anyone else starts their own.
        raise HTTPException(status_code=404, detail=f"thread '{body.thread_id}' not found")
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
    slug: str,
    unread_only: bool = False,
    session: Session = Depends(get_session),
    caller: Caller = Depends(get_caller),
) -> list[AgentMessageOut]:
    agent = a2a.resolve_agent(session, slug)
    if agent is None:
        raise HTTPException(status_code=404, detail=f"agent '{slug}' not found")
    if caller.agent is not None and caller.agent.id != agent.id:
        raise HTTPException(status_code=403, detail="an agent may only read its own inbox")
    return [
        AgentMessageOut.model_validate(m)
        for m in a2a.inbox(session, agent, unread_only=unread_only)
    ]


@router.get("/threads/{thread_id}", response_model=list[AgentMessageOut])
def get_thread(
    thread_id: uuid.UUID,
    session: Session = Depends(get_session),
    caller: Caller = Depends(get_caller),
) -> list[AgentMessageOut]:
    if caller.agent is not None and not a2a.in_thread(session, thread_id, caller.agent):
        raise HTTPException(status_code=404, detail=f"thread '{thread_id}' not found")
    return [AgentMessageOut.model_validate(m) for m in a2a.thread(session, thread_id)]
