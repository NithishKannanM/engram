"""Real Ollama extraction provider (grammar-constrained JSON).

POSTs ``/api/chat`` with ``format`` set to the *flattened* ExtractionResult schema
so decoding is grammar-constrained, ``temperature=0`` + ``seed`` for determinism,
``num_predict`` capped so a constrained small model can't loop forever without
closing the JSON, and ``keep_alive`` so the model stays warm (cold load ~8.5s
dominates cost; warm calls are sub-second).

All local inference is serialized behind one process-wide semaphore: this box has
6GB VRAM shared with Postgres and the API, so concurrent model calls would thrash.
"""

from __future__ import annotations

import time
from collections.abc import Sequence

import httpx

from local_infra.config import get_settings
from local_infra.extraction.providers.base import ProviderResult, Stage
from local_infra.extraction.schemas import ExtractionResult, flatten_schema
from local_infra.llm.gpu import OLLAMA_LOCK as _OLLAMA_LOCK

# _OLLAMA_LOCK is imported, not defined here, so the chat router shares the SAME
# semaphore: small/repair/large AND chat all contend for one 6GB GPU. Defining a
# second one anywhere would give both subsystems a lock that guards nothing.

_SYSTEM = (
    "You are a precise knowledge-extraction engine. Extract durable facts about the "
    "user or their world as subject-predicate-object triples. For every fact, copy an "
    "`evidence_span` VERBATIM from the source text that supports it — do not paraphrase "
    "it. Prefer stable predicates like works_at, lives_in, prefers_editor, knows_language. "
    "Set `confidence` to your own certainty (0..1). If nothing durable is stated, return "
    "an empty `facts` array. Output ONLY JSON matching the schema."
)


class OllamaExtractionProvider:
    def __init__(
        self,
        stage: Stage,
        model: str,
        *,
        base_url: str | None = None,
        repair: bool = False,
        keep_alive: str = "10m",
        num_predict: int = 1536,
        timeout: float = 180.0,
    ) -> None:
        self.stage = stage
        self.model = model
        self._repair = repair
        self._keep_alive = keep_alive
        self._num_predict = num_predict
        self._base_url = (base_url or get_settings().ollama_base_url).rstrip("/")
        self._format = flatten_schema(ExtractionResult)
        self._client = httpx.AsyncClient(timeout=timeout)

    def _user_prompt(self, text: str, focus: Sequence[str] | None) -> str:
        if focus:
            spans = "\n".join(f"- {s}" for s in focus)
            head = (
                "A previous attempt could not ground these spans. Re-extract facts using "
                "ONLY the following spans, copying each evidence_span verbatim:\n"
                f"{spans}\n\nSOURCE:\n"
            )
        elif self._repair:
            head = (
                "A previous extraction was malformed or ungrounded. Extract again carefully, "
                "copying each evidence_span verbatim from the source.\n\nSOURCE:\n"
            )
        else:
            head = "Extract facts from the following source.\n\nSOURCE:\n"
        return head + text

    async def extract(
        self, text: str, *, focus: Sequence[str] | None = None, seed: int = 0
    ) -> ProviderResult:
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": self._user_prompt(text, focus)},
            ],
            "format": self._format,
            "stream": False,
            "keep_alive": self._keep_alive,
            "options": {"temperature": 0, "seed": seed, "num_predict": self._num_predict},
        }
        t0 = time.perf_counter()
        try:
            async with _OLLAMA_LOCK:
                resp = await self._client.post(f"{self._base_url}/api/chat", json=body)
            resp.raise_for_status()
            content = resp.json()["message"]["content"]
        except (httpx.HTTPError, KeyError, ValueError) as exc:
            return ProviderResult(
                raw=None, ok=False, error=f"{type(exc).__name__}: {exc}",
                latency_ms=int((time.perf_counter() - t0) * 1000),
            )
        return ProviderResult(
            raw=content, ok=True, latency_ms=int((time.perf_counter() - t0) * 1000)
        )

    async def close(self) -> None:
        await self._client.aclose()
