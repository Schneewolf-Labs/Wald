"""RAG synthesis over an OpenAI-compatible chat endpoint.

Like the embedding tests, exercised against a real HTTP server rather than a mocked httpx,
so the assertions are about what a server actually receives. None of this needs Postgres:
`search` is replaced at the module boundary where `ask` is involved.
"""

from __future__ import annotations

import json
import threading
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from wald.config import Settings
from wald.schemas import SearchHit
from wald.services import rag

received: list[dict] = []
reply: dict = {}


class _Handler(BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        received.append(
            {"path": self.path, "body": body, "auth": self.headers.get("Authorization")}
        )
        status = reply.get("status", 200)
        payload = json.dumps(
            reply.get("json", {"choices": [{"message": {"content": reply.get("content", "ok")}}]})
        ).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        pass


@pytest.fixture
def server():
    received.clear()
    reply.clear()
    httpd = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_port}/v1"
    httpd.shutdown()
    httpd.server_close()


HITS = [
    SearchHit(
        source_type="wiki", source_id=uuid.uuid4(), title="VPN", snippet="Use WireGuard.", score=1
    )
]


def test_posts_a_chat_completion_with_system_and_user(server):
    reply["content"] = "Use WireGuard [1]."
    answer = rag.synthesize(Settings(llm_base_url=server), "How do I VPN?", HITS)

    assert answer == "Use WireGuard [1]."
    req = received[0]
    assert req["path"] == "/v1/chat/completions"
    assert [m["role"] for m in req["body"]["messages"]] == ["system", "user"]
    assert "[1] (wiki: VPN)\nUse WireGuard." in req["body"]["messages"][1]["content"]
    assert "Question: How do I VPN?" in req["body"]["messages"][1]["content"]
    assert req["body"]["max_tokens"] == 1024
    assert req["body"]["stream"] is False


def test_a_trailing_slash_does_not_produce_a_double_slash(server):
    rag.synthesize(Settings(llm_base_url=server + "/"), "q", HITS)
    assert received[0]["path"] == "/v1/chat/completions"


def test_model_is_sent_only_when_set(server):
    rag.synthesize(Settings(llm_base_url=server), "q", HITS)
    assert "model" not in received[-1]["body"]

    rag.synthesize(Settings(llm_base_url=server, llm_model="qwen3"), "q", HITS)
    assert received[-1]["body"]["model"] == "qwen3"


def test_api_key_is_sent_only_when_set(server):
    rag.synthesize(Settings(llm_base_url=server), "q", HITS)
    assert received[-1]["auth"] is None

    rag.synthesize(Settings(llm_base_url=server, llm_api_key="sk-test"), "q", HITS)
    assert received[-1]["auth"] == "Bearer sk-test"


def test_think_blocks_are_stripped(server):
    reply["content"] = "<think>\nThe user wants VPN.\n</think>\n\nUse WireGuard [1]."
    assert rag.synthesize(Settings(llm_base_url=server), "q", HITS) == "Use WireGuard [1]."


def test_an_unterminated_think_block_is_stripped(server):
    reply["content"] = "<think>ran out of tokens mid-thought"
    assert rag.synthesize(Settings(llm_base_url=server), "q", HITS) == ""


def test_http_errors_name_the_setting(server):
    reply["status"] = 503
    reply["json"] = {"error": "loading model"}
    with pytest.raises(RuntimeError, match="WALD_LLM_BASE_URL .* returned HTTP 503"):
        rag.synthesize(Settings(llm_base_url=server), "q", HITS)


def test_a_malformed_response_names_the_setting(server):
    reply["json"] = {"nope": True}
    with pytest.raises(RuntimeError, match="WALD_LLM_BASE_URL .* OpenAI-shaped"):
        rag.synthesize(Settings(llm_base_url=server), "q", HITS)


def test_an_unreachable_endpoint_names_the_setting():
    with pytest.raises(RuntimeError, match="could not reach .* WALD_LLM_BASE_URL"):
        rag.synthesize(Settings(llm_base_url="http://127.0.0.1:1/v1"), "q", HITS)


# --- Provider selection ----------------------------------------------------
def test_a_base_url_wins_over_anthropic(server, monkeypatch):
    def _no_anthropic(*args, **kwargs):
        raise AssertionError("Anthropic path used despite WALD_LLM_BASE_URL")

    monkeypatch.setattr(rag, "_synthesize_anthropic", _no_anthropic)
    rag.synthesize(Settings(llm_base_url=server, anthropic_api_key="sk-ant"), "q", HITS)
    assert len(received) == 1


def test_anthropic_is_used_when_only_a_key_is_set(monkeypatch):
    monkeypatch.setattr(rag, "_synthesize_anthropic", lambda settings, user: "from claude")
    assert rag.synthesize(Settings(anthropic_api_key="sk-ant"), "q", HITS) == "from claude"


def test_has_llm_reflects_either_option():
    assert not Settings().has_llm
    assert Settings(llm_base_url="http://localhost:1/v1").has_llm
    assert Settings(anthropic_api_key="sk-ant").has_llm


def test_ask_synthesizes_over_the_endpoint(server, monkeypatch):
    reply["content"] = "<think>hm</think>Use WireGuard [1]."
    monkeypatch.setattr(rag, "get_settings", lambda: Settings(llm_base_url=server))
    monkeypatch.setattr(rag, "search", lambda session, q, top_k: HITS)

    resp = rag.ask(None, "How do I VPN?")  # type: ignore[arg-type]
    assert resp.synthesized
    assert resp.answer == "Use WireGuard [1]."
    assert resp.citations == HITS


def test_dev_mode_message_mentions_both_options(monkeypatch):
    monkeypatch.setattr(rag, "get_settings", lambda: Settings())
    monkeypatch.setattr(rag, "search", lambda session, q, top_k: HITS)

    resp = rag.ask(None, "q")  # type: ignore[arg-type]
    assert not resp.synthesized
    assert "WALD_LLM_BASE_URL" in resp.answer
    assert "ANTHROPIC_API_KEY" in resp.answer
