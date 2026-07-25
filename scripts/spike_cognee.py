"""M0.5 spike: can cognee 1.4.0 be driven entirely by OUR infrastructure?

Three questions, answered independently so a partial failure is still informative:

  Q1  relational + vector stores -> our Podman Postgres (not site-packages SQLite/LanceDB)
  Q2  LLM + embeddings -> our Ollama (no cloud, no API key)
  Q3  remember() -> recall() round-trip actually returns the planted fact

If Q1 or Q2 fails, CogneeBackend degrades to best-effort graph recall and
ExMemBackend becomes primary. That is survivable; discovering it at M8 is not.

Run: uv run python scripts/spike_cognee.py
"""

import asyncio
import os
import pathlib
import sys
import traceback

# Must be set before importing cognee: it reads these at module import time.
# Local single-user service -- no tenants, no auth, no session cache in the spike.
os.environ.setdefault("ENABLE_BACKEND_ACCESS_CONTROL", "false")
os.environ.setdefault("CACHING", "false")
# cognee probes the LLM with instructor structured output and a 30s ceiling. On a
# cold Ollama model (8.5s load) plus max_completion_tokens=16384 that intermittently
# trips even though the endpoint is fine -- verified reachable via litellm directly.
os.environ.setdefault("COGNEE_SKIP_CONNECTION_TEST", "true")

# Keep cognee's own state out of .venv/site-packages, where a reinstall would erase it.
STATE = pathlib.Path(__file__).resolve().parent.parent / ".cognee_state"
STATE.mkdir(exist_ok=True)

import cognee  # noqa: E402

PG = dict(
    db_provider="postgres",
    db_host="127.0.0.1",
    db_port="5433",
    db_username="local_infra",
    db_password="local_infra",
    db_name="local_infra",
)
OLLAMA = "http://127.0.0.1:11434"

results: dict[str, tuple[bool, str]] = {}


def record(key: str, ok: bool, detail: str = "") -> None:
    results[key] = (ok, detail)
    print(f"  [{'PASS' if ok else 'FAIL'}] {key}" + (f" -- {detail}" if detail else ""))


async def main() -> int:
    print("\n=== Q1: point relational + vector at our Postgres ===")
    try:
        cognee.config.system_root_directory(str(STATE / "system"))
        cognee.config.data_root_directory(str(STATE / "data"))
        cognee.config.set_relational_db_config(PG)
        record("relational->postgres", True)
    except Exception as e:
        record("relational->postgres", False, f"{type(e).__name__}: {e}")

    try:
        # Vector config accepts ONLY vector_db_*-prefixed keys -- it does not share
        # the relational config's `db_*` shape.
        cognee.config.set_vector_db_config(
            {
                "vector_db_provider": "pgvector",
                "vector_db_url": "",
                "vector_db_key": "",
                "vector_db_host": PG["db_host"],
                "vector_db_port": int(PG["db_port"]),
                "vector_db_name": PG["db_name"],
                "vector_db_username": PG["db_username"],
                "vector_db_password": PG["db_password"],
            }
        )
        record("vector->pgvector", True)
    except Exception as e:
        record("vector->pgvector", False, f"{type(e).__name__}: {e}")

    # Graph: kuzu is the embedded local default. Postgres-backed graph may not exist;
    # find out which providers are actually accepted rather than assuming.
    for provider in ("kuzu", "networkx"):
        try:
            cognee.config.set_graph_database_provider(provider)
            record(f"graph->{provider}", True)
            break
        except Exception as e:
            record(f"graph->{provider}", False, f"{type(e).__name__}: {e}")

    print("\n=== Q2: point LLM + embeddings at our Ollama ===")
    try:
        # cognee's ollama adapter is NOT litellm-routed: it builds
        # AsyncOpenAI(base_url=endpoint) + instructor json_mode. So the endpoint must
        # be the OpenAI-compatible base (/v1) and the model a bare tag -- a
        # litellm-style "ollama_chat/" prefix 404s against /v1/chat/completions.
        cognee.config.set_llm_provider("ollama")
        cognee.config.set_llm_endpoint(f"{OLLAMA}/v1")
        cognee.config.set_llm_model("llama3.1:8b")
        cognee.config.set_llm_api_key("ollama")  # placeholder; litellm requires non-empty
        record("llm->ollama", True)
    except Exception as e:
        record("llm->ollama", False, f"{type(e).__name__}: {e}")

    try:
        cognee.config.set_embedding_provider("ollama")
        # OllamaEmbeddingEngine posts an `input` payload to /api/embed (plural-input
        # API). The older /api/embeddings takes `prompt` instead and silently returns
        # no vector for this payload -- surfacing later as "expected 768 dimensions,
        # not 0" from the pgvector INSERT.
        cognee.config.set_embedding_endpoint(f"{OLLAMA}/api/embed")
        cognee.config.set_embedding_model("nomic-embed-text")
        cognee.config.set_embedding_dimensions(768)
        cognee.config.set_embedding_api_key("ollama")
        record("embeddings->ollama", True)
    except Exception as e:
        record("embeddings->ollama", False, f"{type(e).__name__}: {e}")

    print("\n=== Q3: remember() -> recall() round-trip ===")
    try:
        await cognee.remember("Nithish works at Globex as a backend engineer.")
        record("remember()", True)
    except Exception as e:
        record("remember()", False, f"{type(e).__name__}: {e}")
        traceback.print_exc(limit=3)
        return report()

    try:
        hits = await cognee.recall("Where does Nithish work?")
        text = str(hits)
        found = "globex" in text.lower()
        record("recall() returns planted fact", found, f"{len(text)} chars; globex={found}")
        print(f"\n  raw recall output (first 600 chars):\n  {text[:600]}\n")
    except Exception as e:
        record("recall()", False, f"{type(e).__name__}: {e}")
        traceback.print_exc(limit=3)

    return report()


def report() -> int:
    print("\n=== SPIKE VERDICT ===")
    failed = [k for k, (ok, _) in results.items() if not ok]
    for k, (ok, detail) in results.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {k}{'  ' + detail if detail else ''}")
    if failed:
        print(f"\n  {len(failed)} failed: {', '.join(failed)}")
        print("  -> CogneeBackend may degrade; ExMemBackend becomes primary.")
        return 1
    print("\n  All green: cognee runs on our Postgres + Ollama, fully offline.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
