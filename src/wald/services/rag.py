"""RAG: retrieve across the hub, then synthesize a cited answer.

Synthesis runs on any OpenAI-compatible `/v1` chat endpoint (``WALD_LLM_BASE_URL``) when
one is configured, otherwise on Claude (``ANTHROPIC_API_KEY``), mirroring how embeddings
prefer ``WALD_EMBED_BASE_URL`` over Voyage.
"""

from __future__ import annotations

import re

import httpx
from sqlalchemy.orm import Session

from wald.config import Settings, get_settings
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


# Qwen-style reasoning models served by llama.cpp put their chain of thought inline. An
# unterminated block (the reply was cut off mid-thought) is stripped to the end.
_THINK = re.compile(r"<think>.*?(?:</think>|\Z)", re.DOTALL)


def _strip_think(text: str) -> str:
    return _THINK.sub("", text).strip()


def _synthesize_openai(settings: Settings, user: str) -> str:
    """POST to `{base}/chat/completions`; errors name the setting that points there."""
    url = f"{settings.llm_base_url.rstrip('/')}/chat/completions"  # type: ignore[union-attr]
    headers = {"Authorization": f"Bearer {settings.llm_api_key}"} if settings.llm_api_key else {}
    payload: dict[str, object] = {
        "messages": [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": user},
        ],
        "max_tokens": 1024,
        "stream": False,
    }
    # Servers that host exactly one model reject an unknown `model`; servers that host
    # several require it. Sending it only when set satisfies both.
    if settings.llm_model:
        payload["model"] = settings.llm_model
    try:
        # Generous: a local model on a busy GPU is not a hosted API.
        resp = httpx.post(url, headers=headers, json=payload, timeout=300.0)
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"] or ""
    except httpx.HTTPStatusError as exc:
        raise RuntimeError(
            f"LLM endpoint at WALD_LLM_BASE_URL ({url}) returned HTTP "
            f"{exc.response.status_code}: {exc.response.text[:200]}"
        ) from exc
    except httpx.HTTPError as exc:
        raise RuntimeError(
            f"could not reach the LLM endpoint at WALD_LLM_BASE_URL ({url}): {exc}"
        ) from exc
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise RuntimeError(
            f"LLM endpoint at WALD_LLM_BASE_URL ({url}) did not return an OpenAI-shaped "
            "chat completion (expected choices[0].message.content)"
        ) from exc
    return _strip_think(content)


def _synthesize_anthropic(settings: Settings, user: str) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    response = client.messages.create(
        model=settings.answer_model,
        max_tokens=1024,
        thinking={"type": "adaptive"},
        system=_SYSTEM,
        messages=[{"role": "user", "content": user}],
    )
    return next((b.text for b in response.content if b.type == "text"), "")


def synthesize(settings: Settings, question: str, hits: list[SearchHit]) -> str:
    context = _format_context(hits) or "(no relevant context found)"
    user = f"Context:\n{context}\n\nQuestion: {question}"
    if settings.llm_base_url:
        return _synthesize_openai(settings, user)
    return _synthesize_anthropic(settings, user)


def ask(session: Session, question: str, top_k: int = 6) -> AskResponse:
    settings = get_settings()
    hits = search(session, question, top_k=top_k)

    if not settings.has_llm:
        # Dev mode: return retrieved context without LLM synthesis.
        answer = (
            "LLM synthesis is disabled (set WALD_LLM_BASE_URL to an OpenAI-compatible /v1 "
            "endpoint, or ANTHROPIC_API_KEY). Returning the top retrieved context so the "
            "pipeline is observable:\n\n" + _format_context(hits)
        )
        return AskResponse(question=question, answer=answer, citations=hits, synthesized=False)

    answer = synthesize(settings, question, hits)
    return AskResponse(question=question, answer=answer, citations=hits, synthesized=True)
