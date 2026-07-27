# Rootless Podman + uv. No psql on this box, so migrate/verify go through Python.
.PHONY: up down logs migrate test bench smoke-docs fmt

# podman-compose directly (not `podman compose`) so we don't depend on the
# podman socket / external docker-compose provider being active.
COMPOSE ?= podman-compose

up:
	$(COMPOSE) up -d

down:
	$(COMPOSE) down

logs:
	$(COMPOSE) logs -f postgres

migrate:
	uv run python -m local_infra.backends.pg.migrate

test:
	uv run pytest -q

bench:
	uv run python -m local_infra.bench.cli --backends exmem,cognee --facts 20 --turns 60 --seed 42

# T6 live gate: a real multi-page PDF through pg + pgvector + Ollama embeddings.
# Needs `make up && make migrate` and a running Ollama.
smoke-docs:
	uv run python scripts/documents_smoke.py
