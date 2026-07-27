"""The GPU semaphore must be ONE object, shared by extraction and chat.

This box has 6GB of VRAM and fits one chat-sized model. If extraction and chat each
held their own semaphore, both would report "serialized" while actually running
concurrently — Ollama would thrash, and the symptom (everything mysteriously slow
under load) would be miles from the cause. So identity is asserted, not just presence.
"""

from __future__ import annotations

import asyncio

from local_infra.extraction.providers import ollama as extraction_ollama
from local_infra.llm.gpu import OLLAMA_LOCK


def test_extraction_provider_shares_the_one_gpu_lock() -> None:
    assert extraction_ollama._OLLAMA_LOCK is OLLAMA_LOCK


def test_lock_admits_exactly_one_holder() -> None:
    assert isinstance(OLLAMA_LOCK, asyncio.Semaphore)
    # A second permit would let two generations share 6GB of VRAM.
    assert OLLAMA_LOCK._value == 1


async def test_lock_actually_serializes() -> None:
    order: list[str] = []

    async def worker(name: str) -> None:
        async with OLLAMA_LOCK:
            order.append(f"{name}:enter")
            await asyncio.sleep(0.01)
            order.append(f"{name}:exit")

    await asyncio.gather(worker("a"), worker("b"))

    # No interleaving: each worker's exit precedes the other's enter.
    assert order in (
        ["a:enter", "a:exit", "b:enter", "b:exit"],
        ["b:enter", "b:exit", "a:enter", "a:exit"],
    ), order
