"""RAG: retrieve across the hub, then synthesize a cited answer with Claude."""

from __future__ import annotations

from sqlalchemy.orm import Session

from wald.config import get_settings
from wald.schemas import AskResponse, SearchHit
from wald.services.search import search

_SYSTEM = (
    "You are Wald, the enterprise knowledge assistant. Answer the question using ONLY the "
    "provided context drawn from the company wiki, resource directory, and agent registry. "
    "Cite sources inline as [n] matching the numbered context blocks. If the context does not "
    "contain the answer, say so plainly rather than guessing."
)


def _format_context(hits: list[SearchHit]) -> str:
    blocks = []
    for i, hit in enumerate(hits, start=1):
        label = f"{hit.source_type}: {hit.title}".strip().rstrip(":")
        blocks.append(f"[{i}] ({label})\n{hit.snippet}")
    return "\n\n".join(blocks)


def ask(session: Session, question: str, top_k: int = 6) -> AskResponse:
    settings = get_settings()
    hits = search(session, question, top_k=top_k)

    if not settings.has_llm:
        # Dev mode: return retrieved context without LLM synthesis.
        answer = (
            "LLM synthesis is disabled (no ANTHROPIC_API_KEY). Returning the top retrieved "
            "context so the pipeline is observable:\n\n" + _format_context(hits)
        )
        return AskResponse(question=question, answer=answer, citations=hits, synthesized=False)

    import anthropic

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    context = _format_context(hits) or "(no relevant context found)"
    response = client.messages.create(
        model=settings.answer_model,
        max_tokens=1024,
        thinking={"type": "adaptive"},
        system=_SYSTEM,
        messages=[
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}",
            }
        ],
    )
    answer = next((b.text for b in response.content if b.type == "text"), "")
    return AskResponse(question=question, answer=answer, citations=hits, synthesized=True)
