"""Hybrid search.

The arm most worth testing is the lexical one, because its failure mode is invisible:
matching on the whole query string as a substring means any multi-word question returns
nothing, hybrid search quietly degrades to semantic-only, and every query still returns
plausible-looking results. Nothing errors. These tests assert the lexical arm on its own,
so a regression shows up as a failure rather than as slightly worse rankings.
"""

from __future__ import annotations

import pytest

from wald.models import Embedding
from wald.services.search import _lexical_ranking, _tsquery_terms, search

DOCS = [
    ("wiki", "Deploy process", "Every service ships from main. Roll back by re-deploying the previous SHA."),
    ("wiki", "Incident response", "Page the on-call engineer. Declare an incident in the channel."),
    ("resource", "Analytics Warehouse", "Columnar warehouse at warehouse.internal.acme.example port 5439."),
    ("resource", "Orders API", "Internal REST API for order lookup and fulfilment status."),
]


@pytest.fixture
def corpus(session):
    from wald.services.embeddings import get_embedding_provider

    provider = get_embedding_provider()
    import uuid as _uuid

    for source_type, title, body in DOCS:
        text = f"{title}\n\n{body}"
        session.add(
            Embedding(
                source_type=source_type,
                source_id=_uuid.uuid4(),
                chunk_index=0,
                content=text,
                embedding=provider.embed([text])[0],
                meta={"title": title},
            )
        )
    session.flush()


# --- Query construction (no database) --------------------------------------
def test_terms_are_ored_not_anded():
    # ANDing a natural-language question matches nothing; this was the actual bug.
    assert _tsquery_terms("how do we ship code") == "how | do | we | ship | code"


def test_tsquery_syntax_in_user_input_is_stripped():
    # `&`, `|`, `!`, `:` and parentheses are tsquery operators. Passing them through would
    # produce a syntax error at best and a query the user did not write at worst.
    assert _tsquery_terms("deploy & (rollback | !prod):*") == "deploy | rollback | prod"


def test_a_query_with_no_words_yields_nothing():
    assert _tsquery_terms("!!! ???") == ""
    assert _tsquery_terms("") == ""


# --- Lexical arm (needs Postgres) ------------------------------------------
def test_multi_word_question_finds_the_right_page(session, corpus):
    hits = _lexical_ranking(session, "how do we ship code to production", 10)
    assert hits, "a multi-word question must reach the lexical arm at all"
    assert hits[0].meta["title"] == "Deploy process"


def test_ranking_prefers_more_matching_terms(session, corpus):
    hits = _lexical_ranking(session, "incident on-call engineer", 10)
    assert hits[0].meta["title"] == "Incident response"


def test_stemming_matches_inflections(session, corpus):
    # "deploying" must find "re-deploying"/"Deploy"; a substring match would not.
    hits = _lexical_ranking(session, "deploying services", 10)
    assert {h.meta["title"] for h in hits} >= {"Deploy process"}


def test_an_exact_identifier_is_found(session, corpus):
    # The case semantic search is worst at, and the reason to keep a lexical arm at all.
    hits = _lexical_ranking(session, "warehouse.internal.acme.example", 10)
    assert hits[0].meta["title"] == "Analytics Warehouse"


def test_query_of_only_stopwords_returns_nothing_rather_than_erroring(session, corpus):
    # Postgres yields an empty tsquery here; it must not raise.
    assert _lexical_ranking(session, "how do we", 10) == []


def test_nonsense_query_matches_nothing_lexically(session, corpus):
    assert _lexical_ranking(session, "zzzznonexistentterm", 10) == []


# --- Fusion ----------------------------------------------------------------
def test_search_returns_hits_and_respects_top_k(session, corpus):
    hits = search(session, "rolling back a bad deploy", top_k=2)
    assert len(hits) == 2
    assert hits[0].score >= hits[1].score
    assert hits[0].snippet


def test_a_chunk_found_by_both_arms_outranks_one_found_by_either(session, corpus):
    # The point of fusion: agreement between two independent rankings is evidence.
    hits = search(session, "deploy process roll back", top_k=4)
    titles = [h.title for h in hits]
    assert titles[0] == "Deploy process"


def test_search_survives_a_query_with_no_usable_terms(session, corpus):
    # Semantic still answers; lexical contributes nothing. Must not raise.
    hits = search(session, "???", top_k=3)
    assert isinstance(hits, list)
