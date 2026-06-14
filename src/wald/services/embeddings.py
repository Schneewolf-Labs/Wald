"""Embedding providers.

Production path: Voyage AI (Anthropic's recommended embedding provider).
Dev path: a deterministic local hash embedding, so the whole stack runs with no
API keys. Both return ``embed_dim``-length unit-ish vectors.
"""

from __future__ import annotations

import hashlib
import math
from typing import Protocol

import httpx

from wald.config import Settings, get_settings


class EmbeddingProvider(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class HashEmbeddingProvider:
    """Deterministic, dependency-free embeddings for local development.

    Not semantically meaningful — it only lets the pgvector plumbing and the
    end-to-end flow run without external services. Swap in Voyage for real use.
    """

    def __init__(self, dim: int) -> None:
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]

    def _embed_one(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for token in text.lower().split():
            h = int(hashlib.sha256(token.encode()).hexdigest(), 16)
            vec[h % self.dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


class VoyageEmbeddingProvider:
    """Calls the Voyage AI embeddings API over HTTP."""

    def __init__(self, api_key: str, model: str) -> None:
        self.api_key = api_key
        self.model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        resp = httpx.post(
            "https://api.voyageai.com/v1/embeddings",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": self.model, "input": texts},
            timeout=30.0,
        )
        resp.raise_for_status()
        data = resp.json()["data"]
        return [item["embedding"] for item in data]


def get_embedding_provider(settings: Settings | None = None) -> EmbeddingProvider:
    settings = settings or get_settings()
    if settings.has_embeddings:
        return VoyageEmbeddingProvider(settings.voyage_api_key, settings.embed_model)  # type: ignore[arg-type]
    return HashEmbeddingProvider(settings.embed_dim)
