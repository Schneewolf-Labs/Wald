"""FastAPI application: the core REST surface (humans + web UI)."""

from __future__ import annotations

from fastapi import FastAPI

from wald import web
from wald.api import agents, resources, search, wiki
from wald.config import get_settings


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="Wald",
        version="0.0.1",
        summary="Enterprise information hub for AI agents and humans.",
    )

    @app.get("/health", tags=["meta"])
    def health() -> dict[str, object]:
        return {
            "status": "ok",
            "env": settings.env,
            "llm_synthesis": settings.has_llm,
            "voyage_embeddings": settings.has_embeddings,
        }

    app.include_router(wiki.router)
    app.include_router(resources.router)
    app.include_router(agents.router)
    app.include_router(search.router)
    # Last, so the JSON API keeps ownership of the paths it already defines; the web UI
    # lives at / and under /ui and cannot shadow them.
    app.include_router(web.router)
    return app


app = create_app()


def run() -> None:
    """Console-script entry point (``wald-api``)."""
    import uvicorn

    from wald.db import SchemaMismatch, check_embedding_dim, engine

    settings = get_settings()
    # As in the MCP server: grants attach to identities the token layer has proved, so
    # enforcing them without authentication would look locked and not be.
    if settings.enforce_authz and not settings.require_auth:
        raise SystemExit(
            "wald-api: WALD_ENFORCE_AUTHZ requires WALD_REQUIRE_AUTH -- authorization "
            "without authentication would enforce grants against unverified identities"
        )

    # Same reasoning as the MCP server: a dimension mismatch makes every search fail while
    # the service looks healthy, so refuse to start instead.
    try:
        check_embedding_dim(engine)
    except SchemaMismatch as exc:
        raise SystemExit(f"wald-api: {exc}") from exc

    uvicorn.run(
        "wald.main:app", host=settings.host, port=settings.port, reload=settings.env == "dev"
    )
