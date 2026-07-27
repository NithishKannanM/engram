"""Value types for the chat layer.

Plain dataclasses, matching the convention the rest of the service follows: only
*requests* get pydantic models (``api/schemas.py``); responses are the service's own
dataclasses, which FastAPI's ``jsonable_encoder`` serializes. One source of truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal
from urllib.parse import urlparse

Role = Literal["system", "user", "assistant"]

# Namespaces whose inference runs on this machine.
LOCAL_MODEL_PREFIXES: tuple[str, ...] = ("ollama/", "ollama_chat/")

# Hosts that are unambiguously this machine. Matches the allowlist asserted by
# tests/test_offline_guard.py, deliberately: one definition of "local", not two.
LOCAL_HOSTS: frozenset[str] = frozenset({"127.0.0.1", "::1", "localhost"})


class ChatUnavailable(RuntimeError):
    """Every deployment for an alias failed. Carries what was tried, so the caller
    can report *why* rather than a bare 503."""

    def __init__(self, alias: str, attempts: list[tuple[str, str]]) -> None:
        self.alias = alias
        self.attempts = attempts
        detail = "; ".join(f"{model}: {err}" for model, err in attempts) or "no deployments"
        super().__init__(f"no deployment answered for alias {alias!r} ({detail})")


@dataclass(frozen=True)
class ChatMessage:
    role: Role
    content: str

    def as_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass(frozen=True)
class Deployment:
    """One concrete way to satisfy an alias. Tried in list order."""

    alias: str
    model: str
    api_base: str | None = None
    api_key: str | None = None
    #: Ollama silently truncates to 2048 tokens without this, which looks exactly
    #: like "retrieval is broken" when a RAG prompt gets its context cut off.
    num_ctx: int | None = None

    @property
    def is_local(self) -> bool:
        """True only when BOTH the model namespace and the endpoint are local.

        Both halves matter: ``ollama/llama3.2`` pointed at a remote host is not local,
        and a local ``api_base`` does not make ``anthropic/claude-sonnet-5`` local.
        """
        if not self.model.startswith(LOCAL_MODEL_PREFIXES):
            return False
        if self.api_base is None:
            return True
        host = urlparse(self.api_base).hostname
        return host in LOCAL_HOSTS


@dataclass
class ChatUsage:
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    #: True when pricing could not be established — reported rather than billed as $0.
    cost_unknown: bool = False


@dataclass
class ChatResult:
    text: str
    alias: str
    model: str
    usage: ChatUsage
    #: True when the first deployment failed and a later one answered.
    fallback_used: bool = False
    #: Every model tried, in order — the first entry is the preferred deployment.
    attempts: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class StreamDelta:
    """One incremental chunk of a streamed completion."""

    text: str
