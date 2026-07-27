"""The single serialization point for local GPU inference.

This box has **6GB of VRAM** shared between Ollama, Postgres and the API, and only
one chat-sized model fits at a time. Two concurrent generations do not run twice as
fast — they thrash: Ollama evicts and reloads, and a cold load costs ~8.5s, which is
why the extraction provider bothers with ``keep_alive`` at all.

So every path that makes the GPU generate tokens — the extraction ladder's three
rungs *and* the chat router — must hold this one semaphore. It lives here, rather
than inside the extraction provider where it started, precisely so chat can share
the same object instead of quietly creating a second one that guards nothing.

There is a test (``tests/test_gpu_lock.py``) asserting the extraction provider and
the chat router observe the *identical* object. If that ever fails, the two
subsystems are contending for the GPU without knowing about each other.

Note what this does NOT cover: embeddings go through a synchronous httpx client on a
worker thread, and long streaming generations hold this lock for the whole stream —
callers must release it in a ``finally`` so a disconnected client cannot wedge the GPU.
"""

from __future__ import annotations

import asyncio

# One permit: strictly one local generation in flight, process-wide.
OLLAMA_LOCK = asyncio.Semaphore(1)
