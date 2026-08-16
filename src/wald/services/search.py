"""Hybrid retrieval: lexical (Postgres full-text) + semantic (pgvector), fused via RRF.

The two arms fail in opposite directions, which is the whole reason for running both.
Semantic search finds a page about shipping code when the question said "deploy" but has
no notion of exactness, so a literal identifier -- a hostname, an error code, a service
name -- can rank below something merely topical. Lexical search nails the identifier and
misses the paraphrase entirely. Reciprocal-rank fusion takes rank rather than score from
each, so neither arm's units need to be comparable to the other's.

The lexical arm ORs its terms. A natural-language question ANDed together ("how & do & we
& ship & code") matches nothing, which is the failure that made this arm dead weight: any
multi-word question returned zero lexical hits and hybrid search silently collapsed to
semantic-only. ORing lets `ts_rank` do its job, scoring a chunk that matches four terms
above one that matches one.
"""

from __future__ import annotations

import re

from sqlalchemy import Float, func, select
from sqlalchemy.orm import Session

from wald.models import Embedding
from wald.schemas import SearchHit
from wald.services.embeddings import get_embedding_provider

_RRF_K = 60  # reciprocal-rank-fusion damping constant
_FTS_CONFIG = "english"

# Only word characters survive. Postgres applies its own stemming and stopword removal to
# each term, so this does not need to be clever -- it needs to guarantee that nothing
# reaching to_tsquery can be read as tsquery syntax (`&`, `|`, `!`, `<->`, parentheses).
_WORD = re.compile(r"\w+", re.UNICODE)


def _tsquery_terms(query: str) -> str:
    """Turn free text into an OR'd tsquery string, or "" when there is nothing to match."""
    return " | ".join(_WORD.findall(query))


def _semantic_ranking(session: Session, query: str, limit: int) -> list[Embedding]:
    query_vec = get_embedding_provider().embed([query])[0]
    stmt = select(Embedding).order_by(Embedding.embedding.cosine_distance(query_vec)).limit(limit)
    return list(session.scalars(stmt))


def _lexical_ranking(session: Session, query: str, limit: int) -> list[Embedding]:
    terms = _tsquery_terms(query)
    if not terms:
        return []

    # to_tsvector is computed here rather than stored; init_db creates a matching GIN
    # index on the same expression, so the planner uses it instead of scanning.
    tsvector = func.to_tsvector(_FTS_CONFIG, Embedding.content)
    tsquery = func.to_tsquery(_FTS_CONFIG, terms)
    stmt = (
        select(Embedding)
        .where(tsvector.op("@@")(tsquery))
        .order_by(func.ts_rank(tsvector, tsquery).cast(Float).desc())
        .limit(limit)
    )
    return list(session.scalars(stmt))


def search(session: Session, query: str, top_k: int = 6) -> list[SearchHit]:
    """Return fused, de-duplicated hits across all pillars."""
    pool = max(top_k * 3, 10)
    semantic = _semantic_ranking(session, query, pool)
    lexical = _lexical_ranking(session, query, pool)

    scores: dict[str, float] = {}
    rows: dict[str, Embedding] = {}
    for ranking in (semantic, lexical):
        for rank, emb in enumerate(ranking):
            # Rows come from one session and are identity-mapped, so the same chunk found
            # by both arms is the same object and its contributions add rather than
            # competing as two near-duplicate results.
            key = str(emb.id)
            scores[key] = scores.get(key, 0.0) + 1.0 / (_RRF_K + rank)
            rows[key] = emb

    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    return [
        SearchHit(
            source_type=rows[key].source_type,
            source_id=rows[key].source_id,
            title=rows[key].meta.get("title", "") if isinstance(rows[key].meta, dict) else "",
            snippet=rows[key].content[:300],
            score=round(score, 5),
        )
        for key, score in ordered
    ]
