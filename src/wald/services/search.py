"""Hybrid retrieval: lexical (Postgres) + semantic (pgvector), fused via RRF."""

from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from wald.models import Embedding
from wald.schemas import SearchHit
from wald.services.embeddings import get_embedding_provider

_RRF_K = 60  # reciprocal-rank-fusion damping constant


def _semantic_ranking(session: Session, query: str, limit: int) -> list[Embedding]:
    query_vec = get_embedding_provider().embed([query])[0]
    stmt = (
        select(Embedding)
        .order_by(Embedding.embedding.cosine_distance(query_vec))
        .limit(limit)
    )
    return list(session.scalars(stmt))


def _lexical_ranking(session: Session, query: str, limit: int) -> list[Embedding]:
    # Simple ILIKE prefilter. TODO: upgrade to Postgres full-text (tsvector) or trigram.
    pattern = f"%{query}%"
    stmt = (
        select(Embedding)
        .where(Embedding.content.ilike(pattern))
        .order_by(func.length(Embedding.content))
        .limit(limit)
    )
    return list(session.scalars(stmt))


def search(session: Session, query: str, top_k: int = 6) -> list[SearchHit]:
    """Return fused, de-duplicated hits across all pillars."""
    pool = max(top_k * 3, 10)
    semantic = _semantic_ranking(session, query, pool)
    lexical = _lexical_ranking(session, query, pool)

    scores: dict[int, float] = {}
    rows: dict[int, Embedding] = {}
    for ranking in (semantic, lexical):
        for rank, emb in enumerate(ranking):
            key = id(emb) if emb.id is None else emb.id.int
            scores[key] = scores.get(key, 0.0) + 1.0 / (_RRF_K + rank)
            rows[key] = emb

    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    hits: list[SearchHit] = []
    for key, score in ordered:
        emb = rows[key]
        hits.append(
            SearchHit(
                source_type=emb.source_type,
                source_id=emb.source_id,
                title=emb.meta.get("title", "") if isinstance(emb.meta, dict) else "",
                snippet=emb.content[:300],
                score=round(score, 5),
            )
        )
    return hits
