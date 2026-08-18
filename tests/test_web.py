"""The human web UI.

The load-bearing test here is the XSS one. Wald has no authentication yet, so anyone who
can reach `POST /wiki` can store markdown that this UI renders -- and Python-Markdown
passes raw HTML through by default. The same mechanism that closes that hole is also what
makes the hub's own pages render correctly, since several of them document literal
`<tool_call>` syntax that an HTML-aware parser would swallow silently.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from wald.db import get_session
from wald.main import create_app
from wald.models import Agent, Resource, WikiPage
from wald.web import plain_text, render_markdown


@pytest.fixture
def client(session):
    app = create_app()
    app.dependency_overrides[get_session] = lambda: session
    return TestClient(app)


@pytest.fixture
def content(session):
    session.add(
        WikiPage(
            slug="runbook",
            title="Runbook",
            space="engineering",
            content="# Runbook\n\nRestart with `systemctl restart wald`.\n",
            tags=["ops"],
        )
    )
    session.add(
        Resource(
            slug="warehouse",
            name="Warehouse",
            kind="database",
            description="The analytics warehouse.",
            connection={"host": "warehouse.internal", "port": 5439},
            auth={"method": "vault", "secret_ref": "vault://kv/warehouse"},
        )
    )
    session.add(Agent(slug="kira", name="Kira", capabilities=["coding"], description="An agent."))
    session.flush()


# --- Rendering -------------------------------------------------------------
def test_raw_html_in_a_page_cannot_execute():
    # Stored XSS, reachable through an unauthenticated POST /wiki.
    out = render_markdown("Hello\n\n<script>alert(1)</script>\n")
    assert "<script>" not in out
    assert "&lt;script&gt;" in out


def test_literal_tool_call_syntax_survives():
    # wiki/training-a-model documents this exact syntax; an HTML-aware parser eats it and
    # the page renders with its most important line missing.
    out = render_markdown("Format: <tool_call><function=name></function></tool_call>\n")
    assert "&lt;tool_call&gt;" in out


def test_code_fences_are_not_double_escaped():
    out = render_markdown("```\n<tool_call>x</tool_call>\n```\n")
    assert "&lt;tool_call&gt;" in out
    assert "&amp;lt;" not in out


def test_tables_and_fences_render():
    out = render_markdown("| a | b |\n| - | - |\n| 1 | 2 |\n")
    assert "<table>" in out


def test_a_leading_title_heading_is_dropped_only_when_asked():
    body = "# Runbook\n\nBody text.\n"
    assert "<h1" not in render_markdown(body, drop_title=True)
    assert "<h1" in render_markdown(body)


def test_only_the_first_heading_is_dropped():
    out = render_markdown("# Title\n\ntext\n\n# Later section\n", drop_title=True)
    assert "Later section" in out


# --- Snippet cleanup -------------------------------------------------------
def test_snippets_lose_markdown_syntax():
    assert plain_text("## Rules\n\n- **Nothing** significant") == "Rules - Nothing significant"


def test_links_become_their_text():
    assert plain_text("See [Operational lessons](/wiki/ops).") == "See Operational lessons."


def test_identifiers_keep_their_underscores():
    # Stripping `_` as emphasis turned Q8_0 into Q80 and n_ctx into nctx. This content is
    # mostly identifiers; mangling one is worse than leaving an underscore in.
    assert plain_text("Wichtel `Q8_0` with `n_ctx` and mtp_num_hidden_layers") == (
        "Wichtel Q8_0 with n_ctx and mtp_num_hidden_layers"
    )


# --- Routes ----------------------------------------------------------------
def test_home_lists_every_pillar(client, content):
    body = client.get("/").text
    assert "Runbook" in body
    assert "Warehouse" in body
    assert "Kira" in body


def test_home_is_fine_with_an_empty_hub(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "wald-seed" in r.text  # tells a new operator what to do next


def test_wiki_page_renders_its_markdown(client, content):
    body = client.get("/ui/wiki/runbook").text
    assert "systemctl restart wald" in body
    assert body.count("Runbook</h1>") == 1  # not once from the template and once from the body


def test_resource_shows_connection_and_the_secret_reference(client, content):
    body = client.get("/ui/resources/warehouse").text
    assert "warehouse.internal" in body
    assert "vault://kv/warehouse" in body
    assert "never the credential" in body


def test_agent_page_renders(client, content):
    assert "coding" in client.get("/ui/agents/kira").text


def test_search_results_link_to_their_pages(client, content, session):
    # A result you cannot click is not a result: a hit carries a source id, and the page it
    # belongs to is addressed by slug.
    from sqlalchemy import select

    from wald.services import ingest

    ingest.index_resource(session, session.scalar(select(Resource)))
    session.flush()

    body = client.get("/ui/search?q=analytics warehouse").text
    assert '/ui/resources/warehouse"' in body


def test_search_without_a_query_explains_itself(client):
    assert "Hybrid retrieval" in client.get("/ui/search").text


def test_a_missing_page_is_a_404(client, content):
    assert client.get("/ui/wiki/nope").status_code == 404
    assert client.get("/ui/resources/nope").status_code == 404
    assert client.get("/ui/agents/nope").status_code == 404


def test_the_json_api_still_owns_its_paths(client, content):
    # The UI mounts at / and /ui precisely so it cannot shadow these.
    assert client.get("/wiki").headers["content-type"].startswith("application/json")
    assert client.get("/search?q=x").headers["content-type"].startswith("application/json")
    assert client.get("/health").json()["status"] == "ok"


def test_the_ui_is_absent_from_the_openapi_schema(client):
    # It is HTML for people; listing it as API surface would be noise for agents.
    paths = client.get("/openapi.json").json()["paths"]
    assert "/ui/search" not in paths
    assert "/wiki" in paths
