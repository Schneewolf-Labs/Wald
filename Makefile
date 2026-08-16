.PHONY: up down sync init-db seed api mcp mcp-http test lint fmt

up:           ## Start Postgres + pgvector
	docker compose up -d

down:         ## Stop Postgres
	docker compose down

sync:         ## Install dependencies
	uv sync --extra dev

init-db:      ## Create database tables + pgvector extension
	uv run wald-init-db

api:          ## Run the core REST API (humans + UI)
	uv run wald-api

seed:         ## Load content (SRC=path, default $WALD_CONTENT_DIR)
	uv run wald-seed $(SRC)

mcp:          ## Run the MCP server over stdio (a local child process)
	uv run wald-mcp

mcp-http:     ## Run the MCP server over HTTP (remote agents)
	uv run wald-mcp --transport streamable-http

test:         ## Run tests
	uv run pytest

lint:         ## Lint
	uv run ruff check src tests

fmt:          ## Format
	uv run ruff format src tests
