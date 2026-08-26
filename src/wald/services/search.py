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

from wald.models import Embedding, Resource, WikiPage
from wald.schemas import SearchHit
from wald.services.authz import Grants
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


def _visible(
    session: Session, candidates: list[tuple[float, Embedding]], grants: Grants
) -> list[tuple[float, Embedding]]:
    """Drop hits the caller's grants do not cover.

    Filtering happens on the candidate list rather than in the retrieval SQL, and *before*
    the top_k cut rather than after -- an agent allowed one space should get its k best
    hits from that space, not k minus however many better-ranked forbidden documents
    happened to outrank them. Agent-registry hits always pass: discovery is what the
    registry is for, and it holds no content beyond what an agent announces about itself.
    """
    wiki_ids = [e.source_id for _, e in candidates if e.source_type == "wiki"]
    resource_ids = [e.source_id for _, e in candidates if e.source_type == "resource"]
    # .all() first: passing the Result straight to dict() would read its .keys() method as
    # "this is a mapping" and fail trying to subscript it.
    spaces = (
        dict(
            session.execute(
                select(WikiPage.id, WikiPage.space).where(WikiPage.id.in_(wiki_ids))
            ).all()
        )
        if wiki_ids
        else {}
    )
    slugs = (
        dict(
            session.execute(
                select(Resource.id, Resource.slug).where(Resource.id.in_(resource_ids))
            ).all()
        )
        if resource_ids
        else {}
    )

    def allowed(emb: Embedding) -> bool:
        if emb.source_type == "wiki":
            return grants.allows("wiki", "read", spaces.get(emb.source_id, ""))
        if emb.source_type == "resource":
            return grants.allows("resource", "read", slugs.get(emb.source_id, ""))
        return True

    return [(score, emb) for score, emb in candidates if allowed(emb)]


def search(
    session: Session, query: str, top_k: int = 6, grants: Grants | None = None
) -> list[SearchHit]:
    """Return fused, de-duplicated hits across all pillars.

    ``grants`` scopes the results to what that caller may read; None means unrestricted
    (the REST surface and the web UI, and the MCP surface while enforcement is off).
    """
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

    # Collapse to one hit per source, keeping its best-scoring chunk. Ranking happens over
    # chunks, so a long wiki page can occupy several of the top slots with different parts
    # of itself -- which is the same answer repeated, at the cost of the other documents
    # the caller asked for. top_k means "k things to look at", not "k fragments".
    best: dict[tuple[str, str], tuple[float, Embedding]] = {}
    for key, score in scores.items():
        emb = rows[key]
        source = (emb.source_type, str(emb.source_id))
        if source not in best or score > best[source][0]:
            best[source] = (score, emb)

    ordered = sorted(best.values(), key=lambda pair: pair[0], reverse=True)
    if grants is not None and not grants.unrestricted:
        ordered = _visible(session, ordered, grants)
    ordered = ordered[:top_k]
    return [
        SearchHit(
            source_type=emb.source_type,
            source_id=emb.source_id,
            title=emb.meta.get("title", "") if isinstance(emb.meta, dict) else "",
            snippet=emb.content[:300],
            score=round(score, 5),
        )
        for score, emb in ordered
    ]
