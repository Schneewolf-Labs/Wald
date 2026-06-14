.PHONY: up down sync init-db api mcp test lint fmt

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

mcp:          ## Run the MCP server (agents)
	uv run wald-mcp

test:         ## Run tests
	uv run pytest

lint:         ## Lint
	uv run ruff check src tests

fmt:          ## Format
	uv run ruff format src tests
