"""M5 verification: the extraction ladder, driven by scripted fake providers.

Asserts exact stage sequences for scripted failures, per-fact partial
re-extraction, the two-rung jump on very-low confidence, complexity fast-path,
and budget-gated cloud. No model, no network.
"""

from __future__ import annotations

import json

from local_infra.extraction.budget import Budget
from local_infra.extraction.ladder import ExtractionLadder
from local_infra.extraction.providers.base import ProviderResult, Stage
from local_infra.extraction.schemas import ExtractionResult, flatten_schema
from tests._fakes import (
    GOOD, GOOD2, HALLUC, MED, SRC,
    FakeProvider, bad_json, empty_result, facts_result, transport_fail,
)


def build(providers, *, offline=False, budget=None):
    return ExtractionLadder(
        {p.stage: p for p in providers}, offline=offline, budget=budget or Budget(1.0)
    )


# --------------------------------------------------------------------------- #
async def test_gate_skips_non_assertion() -> None:
    small = FakeProvider(Stage.LOCAL_SMALL, [facts_result(GOOD)])
    res = await build([small]).run("hello there friend")
    assert res.final_stage is Stage.GATE
    assert res.facts == [] and small.calls == []  # provider never called
    assert res.stages == [Stage.GATE]


async def test_happy_path_local_small() -> None:
    small = FakeProvider(Stage.LOCAL_SMALL, [facts_result(GOOD, GOOD2)])
    res = await build([small]).run(SRC)
    assert res.stages == [Stage.GATE, Stage.LOCAL_SMALL]
    assert res.final_stage is Stage.LOCAL_SMALL
    assert len(res.facts) == 2 and not res.deferred


async def test_transport_failure_escalates_to_repair() -> None:
    small = FakeProvider(Stage.LOCAL_SMALL, [transport_fail()])
    repair = FakeProvider(Stage.LOCAL_REPAIR, [facts_result(GOOD)])
    res = await build([small, repair]).run(SRC)
    assert res.stages == [Stage.GATE, Stage.LOCAL_SMALL, Stage.LOCAL_REPAIR]
    assert len(res.facts) == 1


async def test_bad_json_escalates() -> None:
    small = FakeProvider(Stage.LOCAL_SMALL, [bad_json()])
    repair = FakeProvider(Stage.LOCAL_REPAIR, [facts_result(GOOD)])
    res = await build([small, repair]).run(SRC)
    assert res.stages == [Stage.GATE, Stage.LOCAL_SMALL, Stage.LOCAL_REPAIR]


async def test_empty_extraction_escalates_when_gate_predicted() -> None:
    small = FakeProvider(Stage.LOCAL_SMALL, [empty_result()])
    repair = FakeProvider(Stage.LOCAL_REPAIR, [facts_result(GOOD)])
    res = await build([small, repair]).run(SRC)
    assert res.stages == [Stage.GATE, Stage.LOCAL_SMALL, Stage.LOCAL_REPAIR]


async def test_ungrounded_fact_jumps_two_rungs_to_large() -> None:
    small = FakeProvider(Stage.LOCAL_SMALL, [facts_result(HALLUC)])
    repair = FakeProvider(Stage.LOCAL_REPAIR, [facts_result(GOOD)])  # must be SKIPPED
    large = FakeProvider(Stage.LOCAL_LARGE, [facts_result(GOOD)])
    res = await build([small, repair, large]).run(SRC)
    assert res.stages == [Stage.GATE, Stage.LOCAL_SMALL, Stage.LOCAL_LARGE]
    assert repair.calls == []  # jumped over
    assert large.calls[0] == [HALLUC["evidence_span"]]  # focused on the failing span


async def test_partial_reextraction_keeps_good_escalates_bad() -> None:
    # SMALL yields one grounded fact (kept) and one hallucination (escalated alone).
    small = FakeProvider(Stage.LOCAL_SMALL, [facts_result(GOOD, HALLUC)])
    large = FakeProvider(Stage.LOCAL_LARGE, [facts_result(GOOD2)])
    res = await build([small, large]).run(SRC)

    assert res.stages == [Stage.GATE, Stage.LOCAL_SMALL, Stage.LOCAL_LARGE]
    assert {f.fact.object for f in res.facts} == {"globex", "chennai"}
    assert large.calls[0] == [HALLUC["evidence_span"]]  # only the bad span re-extracted


async def test_cloud_reached_online_within_budget() -> None:
    small = FakeProvider(Stage.LOCAL_SMALL, [facts_result(MED)])
    repair = FakeProvider(Stage.LOCAL_REPAIR, [facts_result(MED)])
    large = FakeProvider(Stage.LOCAL_LARGE, [facts_result(MED)])
    cloud = FakeProvider(Stage.CLOUD, [facts_result(GOOD, cost=0.01)])
    res = await build([small, repair, large, cloud], budget=Budget(1.0)).run(SRC)
    assert res.stages[-1] is Stage.CLOUD
    assert res.final_stage is Stage.CLOUD
    assert len(res.facts) == 1 and res.cost_usd == 0.01


async def test_cloud_skipped_when_budget_exhausted() -> None:
    small = FakeProvider(Stage.LOCAL_SMALL, [facts_result(MED)])
    repair = FakeProvider(Stage.LOCAL_REPAIR, [facts_result(MED)])
    large = FakeProvider(Stage.LOCAL_LARGE, [facts_result(MED)])
    cloud = FakeProvider(Stage.CLOUD, [facts_result(GOOD, cost=0.01)])
    res = await build([small, repair, large, cloud], budget=Budget(0.0)).run(SRC)
    assert Stage.CLOUD not in res.stages
    assert cloud.calls == []
    assert res.final_stage is Stage.DEFER


async def test_complex_document_fast_paths_to_large() -> None:
    text = "I work at Globex. " * 100  # > 1500 chars, many cues -> complex
    good = dict(subject="i", predicate="works_at", object="globex",
                evidence_span="I work at Globex", confidence=0.9)
    small = FakeProvider(Stage.LOCAL_SMALL, [facts_result(good)])
    repair = FakeProvider(Stage.LOCAL_REPAIR, [facts_result(good)])
    large = FakeProvider(Stage.LOCAL_LARGE, [facts_result(good)])
    res = await build([small, repair, large]).run(text)
    assert res.stages == [Stage.GATE, Stage.LOCAL_LARGE]
    assert small.calls == [] and repair.calls == []


async def test_flatten_schema_inlines_all_refs() -> None:
    flat = flatten_schema(ExtractionResult)

    def has_ref(node) -> bool:
        if isinstance(node, dict):
            return "$ref" in node or any(has_ref(v) for v in node.values())
        if isinstance(node, list):
            return any(has_ref(x) for x in node)
        return False

    assert "$defs" not in flat
    assert not has_ref(flat)
    # The inlined fact shape survived flattening.
    item = flat["properties"]["facts"]["items"]
    assert set(item["properties"]) >= {"subject", "predicate", "object", "evidence_span"}


# --------------------------------------------------------------------------- #
# Blank-object guard.
#
# Asking a QUESTION makes models emit a subject/predicate with an empty object.
# Such a row is still single-valued, so it supersedes the real answer and
# destroys a true fact. It must never survive parsing — but it also must not
# take well-formed facts extracted alongside it down with it.
# --------------------------------------------------------------------------- #
async def test_blank_object_fact_is_dropped_not_stored() -> None:
    raw = json.dumps(
        {"facts": [dict(GOOD, object=""), dict(GOOD, object="   ")]}
    )
    small = FakeProvider(Stage.LOCAL_SMALL, [ProviderResult(raw=raw)])
    repair = FakeProvider(Stage.LOCAL_REPAIR, [facts_result(GOOD)])
    res = await build([small, repair]).run(SRC)

    # Nothing usable at the first rung -> escalate, and no blank ever emitted.
    assert all(f.fact.object.strip() for f in res.facts)
    assert res.facts and res.facts[0].fact.object == "globex"


async def test_blank_fact_does_not_discard_its_valid_siblings() -> None:
    raw = json.dumps({"facts": [dict(GOOD, object=""), GOOD2]})
    small = FakeProvider(Stage.LOCAL_SMALL, [ProviderResult(raw=raw)])
    res = await build([small]).run(SRC)

    objects = [f.fact.object for f in res.facts]
    assert objects == ["chennai"], objects  # the good sibling survived alone
