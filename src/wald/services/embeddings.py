"""Embedding providers.

Three paths, chosen by what is configured:

1. **Any OpenAI-compatible endpoint** (``WALD_EMBED_BASE_URL``) -- a local embedding
   server, vLLM, TEI, llama.cpp, or OpenAI itself.
2. **Voyage AI** (``VOYAGE_API_KEY``).
3. **A deterministic local hash**, so the stack runs with nothing configured at all.

The first exists because the fallback is not a fallback in any useful sense: the hash
embedding is not semantic, so with no key the semantic arm of hybrid search contributes
essentially noise and retrieval quality rests entirely on full-text matching. An
organization running its own models usually already has an embedding server; pointing Wald
at it costs one config line and is the difference between real semantic retrieval and none.

All three return ``embed_dim``-length vectors, and a mismatch is raised rather than
allowed to reach pgvector as a dimension error nobody can read.
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


class OpenAIEmbeddingProvider:
    """Any endpoint speaking the OpenAI `/v1/embeddings` shape.

    Covers a self-hosted embedding server, vLLM, TEI, llama.cpp and OpenAI itself, because
    they all agree on `{"model", "input"} -> {"data": [{"embedding": [...]}]}`.

    Requests are batched. A local server holds every input in memory at once, and a
    re-index of a large wiki in a single request is how you turn a working embedding server
    into an OOM.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        *,
        batch_size: int = 32,
        timeout: float = 120.0,
    ) -> None:
        self.url = f"{base_url.rstrip('/')}/embeddings"
        self.model = model
        self.api_key = api_key
        self.batch_size = batch_size
        # Generous by HTTP standards: a local model on a busy GPU is not a hosted API, and
        # a timeout here surfaces as a failed content load rather than a slow one.
        self.timeout = timeout

    def embed(self, texts: list[str]) -> list[list[float]]:
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        out: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            payload: dict[str, object] = {"input": batch}
            # Servers that host exactly one model reject an unknown `model`; servers that
            # host several require it. Sending it only when set satisfies both.
            if self.model:
                payload["model"] = self.model
            resp = httpx.post(self.url, headers=headers, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()["data"]
            # Order is guaranteed by `index`, not by position in the response.
            out.extend(item["embedding"] for item in sorted(data, key=lambda d: d.get("index", 0)))
        return out


class _DimensionChecked:
    """Wraps a provider and fails loudly when its vectors are the wrong length.

    pgvector rejects a mismatched vector with a message about expected dimensions and no
    hint about which side is wrong. Since `embed_dim` also fixes the column width at table
    creation, the fix is nearly always "the configured dimension does not match the model",
    and saying so directly saves a genuinely confusing debugging session.
    """

    def __init__(self, inner: EmbeddingProvider, dim: int) -> None:
        self.inner = inner
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = self.inner.embed(texts)
        for vector in vectors:
            if len(vector) != self.dim:
                raise ValueError(
                    f"embedding provider returned {len(vector)} dimensions but "
                    f"WALD_EMBED_DIM is {self.dim}. Set WALD_EMBED_DIM to {len(vector)}, "
                    "then recreate the embedding table and re-run wald-seed -- the column "
                    "width is fixed when the table is created."
                )
        return vectors


def get_embedding_provider(settings: Settings | None = None) -> EmbeddingProvider:
    settings = settings or get_settings()
    if settings.embed_base_url:
        return _DimensionChecked(
            OpenAIEmbeddingProvider(
                settings.embed_base_url, settings.embed_model, settings.embed_api_key
            ),
            settings.embed_dim,
        )
    if settings.has_embeddings:
        return VoyageEmbeddingProvider(settings.voyage_api_key, settings.embed_model)  # type: ignore[arg-type]
    return HashEmbeddingProvider(settings.embed_dim)
