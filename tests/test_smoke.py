"""Smoke tests that need neither a database nor API keys."""

from __future__ import annotations

from wald.config import get_settings
from wald.schemas import AskRequest, WikiPageIn
from wald.services.embeddings import HashEmbeddingProvider, get_embedding_provider
from wald.services.ingest import _chunk


def test_app_imports():
    from wald.main import create_app

    app = create_app()
    paths = set(app.openapi()["paths"])
    assert "/health" in paths
    assert "/wiki" in paths
    assert "/ask" in paths
    assert "/agents/messages" in paths


def test_dev_mode_uses_hash_embeddings():
    settings = get_settings()
    assert not settings.has_llm  # no key in test env
    provider = get_embedding_provider(settings)
    assert isinstance(provider, HashEmbeddingProvider)


def test_hash_embedding_is_deterministic_and_correct_dim():
    settings = get_settings()
    provider = HashEmbeddingProvider(settings.embed_dim)
    a = provider.embed(["how do I connect to the warehouse"])[0]
    b = provider.embed(["how do I connect to the warehouse"])[0]
    assert len(a) == settings.embed_dim
    assert a == b


def test_chunking():
    assert _chunk("") == []
    big = "x" * 3000
    chunks = _chunk(big, size=1200)
    assert len(chunks) == 3
    assert "".join(chunks) == big


def test_schemas_validate():
    page = WikiPageIn(slug="onboarding", title="Onboarding", content="welcome")
    assert page.tags == []
    ask = AskRequest(question="what is our deploy process?")
    assert ask.top_k == 6


def test_mcp_server_constructs():
    from wald.mcp.server import mcp

    assert mcp.name == "wald"
