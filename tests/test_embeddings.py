"""Embedding providers.

Exercised against a real HTTP server rather than a mocked httpx, because the things most
likely to break are the wire format and the batching -- and a mock would assert my
assumptions about the request rather than what a server actually receives.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from wald.config import Settings
from wald.services.embeddings import (
    HashEmbeddingProvider,
    OpenAIEmbeddingProvider,
    VoyageEmbeddingProvider,
    _DimensionChecked,
    get_embedding_provider,
)

DIM = 8
received: list[dict] = []


class _Handler(BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        received.append({"body": body, "auth": self.headers.get("Authorization")})
        # Returned out of order on purpose: the OpenAI shape defines order by `index`,
        # and a provider that trusts position will silently mis-pair text to vector.
        items = [{"index": i, "embedding": [float(i)] * DIM} for i in range(len(body["input"]))]
        payload = json.dumps({"data": list(reversed(items))}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        pass


@pytest.fixture
def server():
    received.clear()
    httpd = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_port}/v1"
    httpd.shutdown()
    httpd.server_close()


def test_embeds_over_the_openai_shape(server):
    provider = OpenAIEmbeddingProvider(server, "some-model")
    vectors = provider.embed(["a", "b", "c"])

    assert len(vectors) == 3
    assert all(len(v) == DIM for v in vectors)
    assert received[0]["body"]["input"] == ["a", "b", "c"]
    assert received[0]["body"]["model"] == "some-model"


def test_results_are_ordered_by_index_not_by_position(server):
    # The fixture responds in reverse; the provider must undo that.
    vectors = OpenAIEmbeddingProvider(server, "m").embed(["a", "b", "c"])
    assert [v[0] for v in vectors] == [0.0, 1.0, 2.0]


def test_large_inputs_are_batched(server):
    # A local server holds every input in memory at once; re-indexing a whole wiki in one
    # request is how a working embedding server becomes an OOM.
    provider = OpenAIEmbeddingProvider(server, "m", batch_size=4)
    vectors = provider.embed([str(i) for i in range(10)])

    assert len(vectors) == 10
    assert [len(r["body"]["input"]) for r in received] == [4, 4, 2]


def test_a_trailing_slash_does_not_produce_a_double_slash(server):
    OpenAIEmbeddingProvider(server + "/", "m").embed(["a"])
    assert len(received) == 1  # it reached the server at all


def test_api_key_is_sent_only_when_set(server):
    OpenAIEmbeddingProvider(server, "m").embed(["a"])
    assert received[-1]["auth"] is None

    OpenAIEmbeddingProvider(server, "m", "sk-test").embed(["a"])
    assert received[-1]["auth"] == "Bearer sk-test"


def test_model_is_omitted_when_unset(server):
    # Single-model servers reject an unknown `model`; multi-model servers require it.
    OpenAIEmbeddingProvider(server, "").embed(["a"])
    assert "model" not in received[-1]["body"]


def test_wrong_dimensions_are_reported_against_the_setting(server):
    checked = _DimensionChecked(OpenAIEmbeddingProvider(server, "m"), dim=1024)
    with pytest.raises(ValueError, match="returned 8 dimensions but WALD_EMBED_DIM is 1024"):
        checked.embed(["a"])


# --- Provider selection ----------------------------------------------------
def test_a_base_url_wins_over_voyage():
    settings = Settings(embed_base_url="http://localhost:1/v1", voyage_api_key="key", embed_dim=DIM)
    provider = get_embedding_provider(settings)
    assert isinstance(provider, _DimensionChecked)
    assert isinstance(provider.inner, OpenAIEmbeddingProvider)


def test_voyage_is_used_when_only_a_key_is_set():
    assert isinstance(
        get_embedding_provider(Settings(voyage_api_key="key")), VoyageEmbeddingProvider
    )


def test_nothing_configured_falls_back_to_the_hash():
    assert isinstance(get_embedding_provider(Settings()), HashEmbeddingProvider)
