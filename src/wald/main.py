"""FastAPI application: the core REST surface (humans + web UI)."""

from __future__ import annotations

from fastapi import FastAPI

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
    return app


app = create_app()


def run() -> None:
    """Console-script entry point (``wald-api``)."""
    import uvicorn

    settings = get_settings()
    uvicorn.run("wald.main:app", host=settings.host, port=settings.port, reload=settings.env == "dev")
