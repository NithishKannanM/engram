"""Extraction output schema + schema flattening for Ollama grammar constraint.

Ollama's `format` grammar converter handles ``$ref``/``$defs`` poorly, so
``flatten_schema`` inlines every reference into one shallow object before the
schema is sent. It is a tested function, not an inline hack, because a silent
mis-flatten degrades constrained decoding without any error.
"""

from __future__ import annotations

from copy import deepcopy

from pydantic import BaseModel, Field, field_validator


class ExtractedFact(BaseModel):
    """One extracted assertion. ``evidence_span`` MUST be copied from the source —
    it is the load-bearing input to groundedness scoring."""

    # min_length=1 is load-bearing, not decoration. Asking a QUESTION ("Where does
    # Nithish work?") makes models emit a subject/predicate with an EMPTY object;
    # that degenerate row is still single-valued, so it supersedes the real answer
    # and silently destroys a true fact. Rejecting it at parse time is the only
    # place that protects every caller.
    subject: str = Field(min_length=1)
    predicate: str = Field(min_length=1)
    object: str = Field(min_length=1)
    evidence_span: str = Field(description="verbatim source text supporting this fact")
    confidence: float = Field(default=0.7, description="model's self-reported confidence 0..1")

    @field_validator("subject", "predicate", "object")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("must not be blank")
        return v.strip()


class ExtractionResult(BaseModel):
    facts: list[ExtractedFact] = Field(default_factory=list)


def flatten_schema(model: type[BaseModel]) -> dict:
    """Return ``model``'s JSON schema with all ``$ref``/``$defs`` inlined."""
    schema = model.model_json_schema()
    defs = schema.pop("$defs", {})

    def resolve(node):
        if isinstance(node, dict):
            if "$ref" in node:
                name = node["$ref"].split("/")[-1]
                target = resolve(deepcopy(defs[name]))
                # Keep any sibling keys (e.g. description) alongside the inlined def.
                return {**target, **{k: resolve(v) for k, v in node.items() if k != "$ref"}}
            return {k: resolve(v) for k, v in node.items()}
        if isinstance(node, list):
            return [resolve(x) for x in node]
        return node

    return resolve(schema)
