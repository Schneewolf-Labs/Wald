"""Application configuration, loaded from the environment / .env.

Most settings use the ``WALD_`` prefix (e.g. ``WALD_DATABASE_URL``). The two
provider keys also accept their conventional unprefixed names
(``ANTHROPIC_API_KEY``, ``VOYAGE_API_KEY``).
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="WALD_", env_file=".env", extra="ignore")

    # Database
    database_url: str = "postgresql+psycopg://wald:wald@localhost:5432/wald"

    # LLM / RAG synthesis
    anthropic_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("ANTHROPIC_API_KEY", "WALD_ANTHROPIC_API_KEY"),
    )
    answer_model: str = "claude-opus-5"

    # Embeddings
    voyage_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("VOYAGE_API_KEY", "WALD_VOYAGE_API_KEY"),
    )
    embed_model: str = "voyage-3.5"
    embed_dim: int = 1024

    # Content
    # A directory of files is the source of truth for the hub; see services/seed.py. The
    # default is gitignored, because real tenant content is exactly what must never be
    # committed to a public repo.
    content_dir: str = "content"

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    env: str = "dev"

    # MCP server. stdio suits an agent that spawns Wald as a child process; streamable-http
    # is what a remote agent on another host needs, and most agents are on another host.
    mcp_transport: str = "stdio"
    mcp_host: str = "0.0.0.0"
    mcp_port: int = 8091

    @property
    def has_llm(self) -> bool:
        """True when a real Claude key is configured (otherwise: dev mode)."""
        return bool(self.anthropic_api_key)

    @property
    def has_embeddings(self) -> bool:
        """True when a real Voyage key is configured (otherwise: hash fallback)."""
        return bool(self.voyage_api_key)


@lru_cache
def get_settings() -> Settings:
    return Settings()
