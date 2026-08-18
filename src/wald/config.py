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
    # populate_by_name so the aliased fields (the two provider keys) can also be set by
    # their field name. Without it a validation_alias *replaces* the field name, and
    # `Settings(voyage_api_key=...)` silently does nothing -- which makes the config
    # awkward to construct in a test and surprising everywhere else.
    model_config = SettingsConfigDict(
        env_prefix="WALD_", env_file=".env", extra="ignore", populate_by_name=True
    )

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

    # Any OpenAI-compatible `/v1` endpoint, which takes precedence over Voyage. Set this to
    # a self-hosted embedding server to get real semantic retrieval without an API key --
    # the hash fallback is not semantic, so without one of these the semantic arm of hybrid
    # search contributes noise. Include the `/v1`: e.g. http://127.0.0.1:8082/v1
    embed_base_url: str | None = None
    embed_api_key: str | None = None

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
    #
    # Loopback by default: the MCP surface has no authentication yet, and `from_agent` on
    # a message is a claim rather than a proof. Exposing it beyond the machine should be a
    # decision someone makes, not one they inherit from a default.
    mcp_transport: str = "stdio"
    mcp_host: str = "127.0.0.1"
    mcp_port: int = 8091

    # Agent authentication. Off by default so an existing hub does not lock out every agent
    # the moment it upgrades -- turning it on is a deliberate act, taken once tokens have been
    # issued. With it on, `from_agent` stops being a parameter the caller controls.
    require_auth: bool = False
    # Advertised to clients during the OAuth-style handshake; only meaningful over HTTP.
    auth_issuer_url: str = "http://127.0.0.1:8091"

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
