#!/usr/bin/env python
"""T6 live smoke: a real PDF through the real stack — pg + pgvector + Ollama embeddings.

`tests/test_documents_api.py` proves the plumbing against fakes. This proves the parts
fakes cannot: that uvicorn's lifespan actually builds two asyncpg connections, that the
background worker really drains the queue, that `nomic-embed-text` returns vectors
Postgres accepts into `vector(768)`, and that multipart upload works over the wire.

    make up && make migrate
    uv run python scripts/documents_smoke.py

Requires the cognee (Postgres) backend — documents are 501 on exmem, which this script
asserts as a separate step rather than skipping.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from pdf_fixture import build_pdf  # noqa: E402

HOST = "127.0.0.1"
PORT = int(os.environ.get("MEMORY_API_PORT", "8078"))
BASE = f"http://{HOST}:{PORT}"
SCOPE = f"project:doc_smoke_{os.getpid()}"

def _filler(topic: str, page: int) -> list[str]:
    """Enough prose per page that the document spans several chunks — a 3-line PDF
    fits in ONE chunk, which would make the page-attribution check vacuous."""
    lines: list[str] = []
    for i in range(1, 5):
        lines += [
            f"Section {page}.{i}: {topic} is reviewed each quarter by the platform team,",
            f"and the outcome of review {page}.{i} is recorded in the engineering log so",
            "that the decision can be audited later without asking anyone to remember it.",
        ]
    return lines


#: The rollback material sits on page 4, so a correct offset→page mapping has to do
#: real arithmetic to find it — page 1 would come out right by accident.
PAGES = [
    ["Platform Handbook", *_filler("the deployment process", 1)],
    ["The deployment window is every Tuesday at 14:00 UTC.", *_filler("the window", 2)],
    ["Releases outside the window require an exception.", *_filler("exceptions", 3)],
    [
        "Rollbacks are performed with make rollback.",
        "A rollback always targets the previous released tag.",
        "A rollback never requires an exception.",
        *_filler("the rollback runbook", 4),
    ],
    ["On-call rotation is weekly and starts on Monday.", *_filler("on-call", 5)],
    ["Incident reviews happen within three working days.", *_filler("incidents", 6)],
]
ROLLBACK_PAGE = 4


def fail(message: str) -> None:
    print(f"FAIL: {message}")
    raise SystemExit(1)


def check(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def boot() -> subprocess.Popen:
    env = {
        **os.environ,
        "MEMORY_BACKEND": "cognee",
        "MEMORY_OFFLINE": "1",
        "MEMORY_API_PORT": str(PORT),
    }
    proc = subprocess.Popen(
        [
            "uv", "run", "uvicorn", "local_infra.api.app:app",
            "--host", HOST, "--port", str(PORT), "--log-level", "warning",
        ],
        cwd=ROOT,
        env=env,
    )
    for _ in range(90):
        if proc.poll() is not None:
            fail(f"server exited during startup (code {proc.returncode})")
        try:
            if httpx.get(f"{BASE}/health", timeout=2).status_code == 200:
                return proc
        except httpx.HTTPError:
            pass
        time.sleep(1)
    proc.terminate()
    fail("/health never came up")
    raise AssertionError("unreachable")


def main() -> int:
    pdf = build_pdf(PAGES)
    print(f"== built a {len(pdf)} byte PDF, {len(PAGES)} pages ==")

    server = boot()
    try:
        with httpx.Client(base_url=BASE, timeout=120) as client:
            health = client.get("/health").json()
            print(f"[health] backend={health['backend']} ok={health['ok']}")
            check(health["backend"] == "cognee", "expected the cognee/pg backend")

            # --- 1. negative cases, before anything is stored ---------------- #
            elf = client.post(
                "/documents",
                files={"file": ("payload.txt", b"\x7fELF\x02\x01\x01" + b"\x00" * 64)},
                data={"scope": SCOPE},
            )
            check(elf.status_code == 415, f"ELF should be 415, got {elf.status_code}")
            print("[1] ELF rejected 415")

            empty = client.post(
                "/documents",
                files={"file": ("empty.txt", b"")},
                data={"scope": SCOPE},
            )
            check(empty.status_code == 400, f"empty should be 400, got {empty.status_code}")
            print("[1] empty rejected 400")

            # --- 2. upload -------------------------------------------------- #
            response = client.post(
                "/documents",
                files={"file": ("handbook.pdf", pdf, "application/pdf")},
                data={"scope": SCOPE},
            )
            check(response.status_code == 202, f"upload got {response.status_code}: {response.text}")
            body = response.json()
            document_id = body["document_id"]
            print(f"[2] accepted {document_id} status={body['status']} queued={body['queued']}")
            check(body["kind"] == "pdf", f"expected kind=pdf, got {body['kind']}")

            # --- 3. poll until the worker finishes -------------------------- #
            deadline = time.time() + 180
            document = {}
            while time.time() < deadline:
                document = client.get(f"/documents/{document_id}").json()
                if document["status"] in ("ready", "failed"):
                    break
                time.sleep(1)
            check(
                document.get("status") == "ready",
                f"document did not become ready: {document}",
            )
            print(
                f"[3] ready — {document['chunk_count']} chunks, "
                f"{document['pages']} pages, {document['size_bytes']} bytes"
            )
            check(document["chunk_count"] > 0, "ready with zero chunks")
            check(document["pages"] == len(PAGES), f"page count {document['pages']}")

            chunks = client.get(f"/documents/{document_id}/chunks").json()
            check(len(chunks) > 1, f"expected a multi-chunk document, got {len(chunks)}")
            check(chunks[0]["page_from"] == 1, f"first chunk on page {chunks[0]['page_from']}")
            pages_seen = sorted({c["page_from"] for c in chunks})
            check(
                pages_seen[-1] > 1,
                f"every chunk landed on page 1 — the offset→page mapping is wrong: {pages_seen}",
            )
            # The chunk carrying the rollback text must be attributed to the page it is
            # actually printed on, or the citation sends the reader to the wrong page.
            rollback_chunks = [c for c in chunks if "make rollback" in c["content"]]
            check(bool(rollback_chunks), "no chunk contains the rollback text")
            check(
                any(c["page_from"] == ROLLBACK_PAGE for c in rollback_chunks),
                f"rollback text attributed to page(s) "
                f"{[c['page_from'] for c in rollback_chunks]}, expected {ROLLBACK_PAGE}",
            )
            print(f"[3] {len(chunks)} chunks across pages {pages_seen}")

            # --- 4. retrieval ----------------------------------------------- #
            rag = client.post(
                "/rag", json={"query": "how do rollbacks work", "scope": SCOPE}
            ).json()
            docs = [c for c in rag["citations"] if c["kind"] == "document"]
            check(bool(docs), f"no document citations: {rag}")
            top = docs[0]
            check(
                "rollback" in top["text"].lower(),
                f"top chunk is not about rollbacks: {top['text'][:120]!r}",
            )
            check(top["filename"] == "handbook.pdf", f"filename {top['filename']}")
            check(not rag["unavailable"], f"channels unavailable: {rag['unavailable']}")
            print(
                f"[4] /rag cited {top['filename']} p{top['page_from']} "
                f"chunk#{top['chunk_index']} @[{top['char_start']}:{top['char_end']}]"
            )

            # --- 5. fusion with a memory ------------------------------------ #
            # NOTE the wording. The extraction gate is a deterministic cue matcher, and
            # "Priya owns the deployment window" contains NO cue — it is silently gated
            # out (final_stage="gate", 0 facts), leaving nothing for the memory channel
            # to contribute and making this step pass vacuously. "is" is a cue.
            remembered = client.post(
                "/remember",
                json={
                    "content": "Priya is the owner of the deployment window.",
                    "scope": SCOPE,
                    "extract": "sync",
                },
                timeout=300,
            ).json()
            check(
                remembered["extracted"] > 0,
                f"nothing extracted (stage={remembered['final_stage']}) — the memory "
                "channel would contribute nothing and this check would be vacuous",
            )
            fused = client.post(
                "/rag", json={"query": "deployment window owner", "scope": SCOPE}
            ).json()
            kinds = {c["kind"] for c in fused["citations"]}
            print(f"[5] fused kinds={sorted(kinds)} over {len(fused['citations'])} citations")
            check("document" in kinds, "the document channel contributed nothing")
            check("memory" in kinds, "the memory channel contributed nothing")

            # --- 6. same bytes, another scope, survives this delete --------- #
            other_scope = f"{SCOPE}_other"
            other = client.post(
                "/documents",
                files={"file": ("handbook.pdf", pdf, "application/pdf")},
                data={"scope": other_scope},
            ).json()
            check(
                other["document_id"] != document_id,
                "same bytes in another scope must be a separate document",
            )

            deleted = client.delete(f"/documents/{document_id}")
            check(deleted.status_code == 200, f"delete got {deleted.status_code}")
            check(
                client.get(f"/documents/{document_id}").status_code == 404,
                "deleted document still readable",
            )
            after = client.post("/rag", json={"query": "rollback", "scope": SCOPE}).json()
            check(
                not [c for c in after["citations"] if c["kind"] == "document"],
                "deleted document still cited",
            )
            print("[6] deleted; chunks gone from /rag")

            blob_root = ROOT / ".cognee_state" / "documents"
            survivor = client.get(f"/documents/{other['document_id']}").json()
            blob = blob_root / survivor["sha256"][:2] / survivor["sha256"]
            check(
                blob.is_file(),
                f"blob {blob} was unlinked even though another scope still uses it",
            )
            print(f"[6] blob kept for {other_scope} (shared sha256)")

            client.delete(f"/documents/{other['document_id']}")
            check(not blob.is_file(), f"blob {blob} survived the LAST reference")
            print("[6] blob unlinked once the last document referencing it was deleted")

        print("\nALL GREEN")
        return 0
    finally:
        server.terminate()
        try:
            server.wait(timeout=15)
        except subprocess.TimeoutExpired:
            server.kill()


if __name__ == "__main__":
    raise SystemExit(main())
