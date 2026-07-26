"""The extraction escalation ladder.

Runs the cheapest capable rung first and escalates only the *failing subset* of
facts — you escalate a paragraph, not a conversation, which is what keeps cloud
cost near zero. Escalation fires on: transport/parse/validation failure, an empty
extraction where the gate predicted facts, groundedness below the floor, or a
composite below the accept threshold. A very low composite (<0.5) jumps two rungs.

Strict-offline is enforced structurally, not by convention: ``_max_index`` never
exceeds LOCAL_LARGE when offline, so the loop cannot reach CLOUD even if a cloud
provider were somehow present. Combined with the registry not constructing cloud
providers offline, there is no object to call by accident.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from local_infra.extraction import confidence
from local_infra.extraction.budget import Budget
from local_infra.extraction.gate import GateDecision, gate as default_gate
from local_infra.extraction.providers.base import (
    LADDER,
    ExtractionProvider,
    ProviderResult,
    Stage,
)
from local_infra.extraction.schemas import ExtractedFact, ExtractionResult

# composite thresholds
ACCEPT = 0.75
JUMP_TWO_BELOW = 0.50
# structural penalty by attempt index (0 = first pass).
_STRUCTURAL = {0: 1.0, 1: 0.7}
CLOUD_ESTIMATE_USD = 0.01


@dataclass
class ScoredFact:
    fact: ExtractedFact
    score: confidence.Score
    stage: Stage


@dataclass
class AttemptLog:
    """One provider call — telemetry for the trace table and extraction_attempts."""

    stage: Stage
    ok: bool
    latency_ms: int
    cost_usd: float
    parsed: int
    accepted: int


@dataclass
class LadderResult:
    facts: list[ScoredFact] = field(default_factory=list)  # accepted
    deferred: list[str] = field(default_factory=list)  # spans left unresolved
    stages: list[Stage] = field(default_factory=list)  # sequence visited (incl. GATE)
    attempts: list[AttemptLog] = field(default_factory=list)  # one per provider call
    final_stage: Stage = Stage.GATE
    cost_usd: float = 0.0


class ExtractionLadder:
    def __init__(
        self,
        providers: Mapping[Stage, ExtractionProvider],
        *,
        offline: bool = False,
        budget: Budget | None = None,
        gate=default_gate,
    ) -> None:
        self._providers = dict(providers)
        self._offline = offline
        self._budget = budget or Budget(0.0)
        self._gate = gate
        top = Stage.LOCAL_LARGE
        if not offline and self._providers.get(Stage.CLOUD) is not None:
            top = Stage.CLOUD
        self._max_index = LADDER.index(top)

    def max_stage(self) -> Stage:
        return LADDER[self._max_index]

    @property
    def providers(self) -> dict[Stage, ExtractionProvider]:
        """The wired providers — so an owner can close their transports on shutdown."""
        return self._providers

    async def run(self, text: str, *, source_text: str | None = None, seed: int = 0) -> LadderResult:
        source = source_text if source_text is not None else text
        result = LadderResult(stages=[Stage.GATE])

        decision = self._gate(text)
        if not decision.should_extract:
            result.final_stage = Stage.GATE
            return result

        idx = LADDER.index(Stage.LOCAL_LARGE) if decision.is_complex else 0
        idx = min(idx, self._max_index)
        focus: Sequence[str] | None = None
        attempt = 0

        while True:
            stage = LADDER[idx]
            provider = self._providers.get(stage)

            # Stage unavailable, or cloud with no budget -> defer.
            if provider is None or (stage is Stage.CLOUD and not self._budget.can_spend(CLOUD_ESTIMATE_USD)):
                result.deferred = list(focus) if focus else ["<document>"]
                result.final_stage = Stage.DEFER
                return result

            result.stages.append(stage)
            provider_result = await provider.extract(text, focus=focus, seed=seed)
            result.cost_usd += provider_result.cost_usd
            self._budget.record(provider_result.cost_usd)

            parsed = _parse(provider_result, decision)
            if parsed is None:
                # transport/parse/validation/empty -> escalate one rung.
                result.attempts.append(
                    AttemptLog(stage, False, provider_result.latency_ms, provider_result.cost_usd, 0, 0)
                )
                attempt += 1
                idx += 1
                if idx > self._max_index:
                    result.deferred = list(focus) if focus else ["<document>"]
                    result.final_stage = Stage.DEFER
                    return result
                continue

            structural = _STRUCTURAL.get(attempt, 0.4)
            accepted: list[ScoredFact] = []
            failing: list[ScoredFact] = []
            for f in parsed:
                sc = confidence.score_fact(
                    f.evidence_span, source, structural=structural, self_report=f.confidence
                )
                bucket = accepted if (sc.grounded and sc.composite >= ACCEPT) else failing
                bucket.append(ScoredFact(fact=f, score=sc, stage=stage))

            result.attempts.append(
                AttemptLog(
                    stage, True, provider_result.latency_ms, provider_result.cost_usd,
                    len(parsed), len(accepted),
                )
            )
            result.facts.extend(accepted)
            if not failing:
                result.final_stage = stage
                return result

            # Escalate ONLY the failing subset, focused on their spans.
            attempt += 1
            jump = 2 if min(s.score.composite for s in failing) < JUMP_TWO_BELOW else 1
            idx += jump
            if idx > self._max_index:
                result.deferred = [s.fact.evidence_span for s in failing]
                result.final_stage = Stage.DEFER
                return result
            focus = [s.fact.evidence_span for s in failing]


def _parse(pr: ProviderResult, decision: GateDecision) -> list[ExtractedFact] | None:
    """Return validated facts, or None to signal 'escalate'. An empty extraction
    where the gate predicted facts is a failure, not a success."""
    if not pr.ok or pr.raw is None:
        return None
    try:
        payload = json.loads(pr.raw)
    except (ValueError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None

    # Validate fact-by-fact rather than through ExtractionResult: one malformed
    # entry (typically a blank object from an interrogative input) must not
    # discard the well-formed facts extracted alongside it.
    facts: list[ExtractedFact] = []
    for raw_fact in payload.get("facts") or []:
        try:
            facts.append(ExtractedFact.model_validate(raw_fact))
        except ValueError:
            continue

    if not facts and decision.predicted_facts >= 1:
        return None
    return facts
