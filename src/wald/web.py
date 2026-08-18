"""The human front door.

Wald's premise is that anything an agent can do, a person can do. The agents got MCP
first; this is the other half. It is a client of the same service layer the REST API and
the MCP server use, so there is no second implementation of "what is a search result".

Server-rendered on purpose. The pages are read far more than they are interacted with, the
content is markdown that has to be rendered somewhere anyway, and rendering it on the
server means a wiki page is a real URL that works with no JavaScript -- which matters for
something people will paste into chat and expect to open.

Routes live under `/ui` so they cannot collide with the JSON API, which owns `/wiki`,
`/resources`, `/agents` and `/search` and stays the primary surface.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import markdown
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from wald.db import get_session
from wald.models import Agent, Resource, WikiPage
from wald.services import a2a
from wald.services import search as search_svc

router = APIRouter(tags=["web"], include_in_schema=False)
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))


def _renderer() -> markdown.Markdown:
    """Markdown with raw HTML disabled.

    Two reasons, and the safety one is not even the first. Wald's own pages document tool
    call formats -- literal ``<tool_call><function=...>`` -- and a parser that treats those
    as HTML swallows them silently, so the page renders with its most important line
    missing. Disabling raw HTML makes them render as the text they are.

    It is also the only thing standing between an unauthenticated ``POST /wiki`` and stored
    XSS, since anyone who can write a page could otherwise write a script tag into one.
    Deregistering the handlers is better than escaping the source first, which would
    double-escape anything inside a code fence.
    """
    md = markdown.Markdown(extensions=["tables", "fenced_code", "sane_lists", "toc"])
    md.preprocessors.deregister("html_block")
    md.inlinePatterns.deregister("html")
    return md


_LEADING_H1 = re.compile(r"\A\s*#\s+.*?(?:\n+|\Z)")


def render_markdown(text: str, *, drop_title: bool = False) -> str:
    """Render markdown to HTML, optionally dropping a leading top-level heading.

    A page carries its title in frontmatter *and* usually repeats it as the body's first
    `#` heading, which is the natural way to write the file and renders as the title twice.
    The frontmatter is the authority, so the body's copy is what gives way.
    """
    body = _LEADING_H1.sub("", text or "", count=1) if drop_title else (text or "")
    return _renderer().convert(body)


_MD_NOISE = [
    (re.compile(r"```[^\n]*"), " "),  # fence markers, keep the code
    (re.compile(r"!?\[([^\]]*)\]\([^)]*\)"), r"\1"),  # links and images -> their text
    (re.compile(r"^[ \t]*[#>]+[ \t]*", re.MULTILINE), ""),  # heading and quote markers
    (re.compile(r"^[ \t]*\|.*$", re.MULTILINE), " "),  # table rows: unreadable unaligned
    # Asterisks and backticks only. Underscore emphasis is rare in technical writing and
    # stripping it wrecks the identifiers this content is made of -- `Q8_0` became `Q80`,
    # `n_ctx` became `nctx`. Mangling a hostname is worse than leaving an underscore in.
    (re.compile(r"[*`]{1,3}"), ""),
    (re.compile(r"\s+"), " "),
]


def plain_text(text: str) -> str:
    """Strip markdown syntax for display in a snippet.

    Search stores the raw chunk, which is correct -- the index should hold what the author
    wrote. But a result list rendering `## Working rules - **Nothing significant...**` is
    markup where a human wanted a sentence, so the cleanup belongs here at the point of
    display rather than in what gets indexed.
    """
    out = text or ""
    for pattern, replacement in _MD_NOISE:
        out = pattern.sub(replacement, out)
    return out.strip()


templates.env.filters["plain"] = plain_text


def _counts(session: Session) -> dict[str, int]:
    return {
        "wiki": session.scalar(select(func.count()).select_from(WikiPage)) or 0,
        "resources": session.scalar(select(func.count()).select_from(Resource)) or 0,
        "agents": session.scalar(select(func.count()).select_from(Agent)) or 0,
    }


def _group(items: list[Any], key: str) -> dict[str, list[Any]]:
    """Group into an ordered dict by an attribute, for the faceted index."""
    out: dict[str, list[Any]] = {}
    for item in items:
        out.setdefault(getattr(item, key) or "other", []).append(item)
    return dict(sorted(out.items()))


@router.get("/", response_class=HTMLResponse)
def home(request: Request, session: Session = Depends(get_session)):
    pages = list(session.scalars(select(WikiPage).order_by(WikiPage.title)))
    resources = list(session.scalars(select(Resource).order_by(Resource.name)))
    agents = list(session.scalars(select(Agent).order_by(Agent.name)))
    return templates.TemplateResponse(
        request,
        "home.html",
        {
            "counts": _counts(session),
            "spaces": _group(pages, "space"),
            "kinds": _group(resources, "kind"),
            "agents": agents,
        },
    )


@router.get("/ui/search", response_class=HTMLResponse)
def search(request: Request, q: str = "", session: Session = Depends(get_session)):
    hits = search_svc.search(session, q, top_k=20) if q.strip() else []

    # A hit knows its source id, not its slug, and a result you cannot click is not a
    # result. Resolving them in one query per pillar keeps this off the N+1 path.
    links: dict[str, str] = {}
    for model, prefix in (
        (WikiPage, "/ui/wiki/"),
        (Resource, "/ui/resources/"),
        (Agent, "/ui/agents/"),
    ):
        ids = [h.source_id for h in hits]
        if not ids:
            continue
        for row in session.execute(select(model.id, model.slug).where(model.id.in_(ids))):
            links[str(row.id)] = f"{prefix}{row.slug}"

    return templates.TemplateResponse(
        request, "search.html", {"q": q, "hits": hits, "links": links}
    )


@router.get("/ui/wiki/{slug}", response_class=HTMLResponse)
def wiki_page(request: Request, slug: str, session: Session = Depends(get_session)):
    page = session.scalar(select(WikiPage).where(WikiPage.slug == slug))
    if page is None:
        raise HTTPException(status_code=404, detail=f"wiki page '{slug}' not found")
    return templates.TemplateResponse(
        request,
        "page.html",
        {"page": page, "body": render_markdown(page.content, drop_title=True)},
    )


@router.get("/ui/resources/{slug}", response_class=HTMLResponse)
def resource_page(request: Request, slug: str, session: Session = Depends(get_session)):
    resource = session.scalar(select(Resource).where(Resource.slug == slug))
    if resource is None:
        raise HTTPException(status_code=404, detail=f"resource '{slug}' not found")
    return templates.TemplateResponse(
        request,
        "resource.html",
        {"resource": resource, "description": render_markdown(resource.description)},
    )


@router.get("/ui/agents/{slug}", response_class=HTMLResponse)
def agent_page(request: Request, slug: str, session: Session = Depends(get_session)):
    agent = a2a.resolve_agent(session, slug)
    if agent is None:
        raise HTTPException(status_code=404, detail=f"agent '{slug}' not found")
    inbox = a2a.inbox(session, agent)[:20]
    return templates.TemplateResponse(
        request,
        "agent.html",
        {"agent": agent, "description": render_markdown(agent.description), "inbox": inbox},
    )
