"""Search + RAG REST router."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from wald.api.auth import Caller, get_caller
from wald.db import get_session
from wald.schemas import AskRequest, AskResponse, SearchResults
from wald.services import rag
from wald.services import search as search_svc

router = APIRouter(tags=["search"])


@router.get("/search", response_model=SearchResults)
def search(
    q: str,
    top_k: int = 6,
    session: Session = Depends(get_session),
    caller: Caller = Depends(get_caller),
) -> SearchResults:
    hits = search_svc.search(session, q, top_k=top_k, grants=caller.grants)
    return SearchResults(query=q, hits=hits)


@router.post("/ask", response_model=AskResponse)
def ask(
    body: AskRequest,
    session: Session = Depends(get_session),
    caller: Caller = Depends(get_caller),
) -> AskResponse:
    # Grants scope retrieval, so context the caller may not read never reaches synthesis.
    return rag.ask(session, body.question, top_k=body.top_k, grants=caller.grants)
