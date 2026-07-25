"""Core dataclasses + the MemoryStrategy ABC (vendored from ex_mem/strategies/base.py).

The ABC is the shim target the benchmark rig drives; the M11 bench adapter wraps a
:class:`MemoryService` in this interface so ``BenchmarkRunner`` runs unchanged.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class Turn:
    role: Literal["user", "assistant", "system"]
    content: str
    turn_index: int
    timestamp: float = field(default_factory=time.time)
    token_count: int = 0


@dataclass
class ContextWindow:
    """What gets sent to the LLM for a given turn."""

    turns: list[Turn]
    retrieved_chunks: list[str] = field(default_factory=list)
    summary: str = ""
    total_tokens: int = 0


@dataclass
class StrategyStats:
    strategy_name: str
    total_turns: int = 0
    total_evictions: int = 0
    total_retrieval_calls: int = 0
    total_tokens_active: int = 0
    total_tokens_retrieved: int = 0
    retrieval_latency_ms: list[float] = field(default_factory=list)
    turn_latency_ms: list[float] = field(default_factory=list)
    embedding_calls: int = 0

    @property
    def p50_turn_latency(self) -> float:
        if not self.turn_latency_ms:
            return 0.0
        s = sorted(self.turn_latency_ms)
        return s[len(s) // 2]

    @property
    def p95_turn_latency(self) -> float:
        if not self.turn_latency_ms:
            return 0.0
        s = sorted(self.turn_latency_ms)
        return s[min(int(len(s) * 0.95), len(s) - 1)]


class MemoryStrategy(ABC):
    """All strategies implement this interface."""

    name: str = "base"

    @abstractmethod
    def add_turn(self, role: str, content: str) -> None:
        """Ingest a new conversation turn."""

    @abstractmethod
    def get_context(self, query: str = "") -> ContextWindow:
        """Return the context window to send to the LLM for the next turn."""

    @abstractmethod
    def get_stats(self) -> StrategyStats:
        """Return accumulated performance statistics."""

    @abstractmethod
    def reset(self) -> None:
        """Clear all state for a fresh benchmark run."""
