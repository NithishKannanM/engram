"""Synthetic conversation generator (ported from ex_mem's benchmark/generator.py).

Design, unchanged from the prior art:
  * Plant N distinct facts across the first ~40% of a conversation.
  * Fill the rest with realistic but fact-free filler dialogue (retrieval distractors).
  * After the conversation, issue one recall probe per planted fact.

Every fact carries a ``probe`` (the question) and an ``answer`` (ground truth) so the
runner can drive ``recall(probe)`` and the judge can score the retrieved text against
``answer``. Fully seeded: the same ``seed`` yields byte-identical conversations, so
runs are comparable across backends and reproducible.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Literal

FactCategory = Literal["personal", "professional", "technical", "temporal", "preference"]


@dataclass
class Fact:
    id: str
    category: FactCategory
    statement: str  # e.g. "My name is Alice Chen."
    probe: str  # e.g. "What is the user's name?"
    answer: str  # e.g. "Alice Chen"
    planted_at_turn: int = -1


@dataclass
class ConversationTurn:
    role: Literal["user", "assistant"]
    content: str
    contains_fact_ids: list[str] = field(default_factory=list)


@dataclass
class SyntheticConversation:
    turns: list[ConversationTurn]
    facts: list[Fact]
    probes: list[Fact]  # one probe per fact (kept ordered with `facts`)


# (statement_template, probe, answer_template)
_FACT_BANK: list[tuple[str, str, str]] = [
    # personal
    ("My name is {name}.", "What is the user's full name?", "{name}"),
    ("I live in {city}, {country}.", "What city does the user live in?", "{city}"),
    ("My birthday is {month} {day}.", "When is the user's birthday?", "{month} {day}"),
    ("I have {n} siblings.", "How many siblings does the user have?", "{n}"),
    ("My favourite colour is {colour}.", "What is the user's favourite colour?", "{colour}"),
    # professional
    ("I work at {company} as a {role}.", "Where does the user work and in what role?", "{company} as a {role}"),
    ("My annual salary is ${salary}k.", "What is the user's salary?", "${salary}k"),
    ("I've been at {company2} for {years} years.", "How long has the user been at their company?", "{years} years"),
    ("My manager is {manager}.", "Who is the user's manager?", "{manager}"),
    ("My team has {size} engineers.", "How large is the user's engineering team?", "{size} engineers"),
    # technical
    ("My primary language is {lang}.", "What programming language does the user prefer?", "{lang}"),
    ("I use {editor} as my editor.", "What editor does the user use?", "{editor}"),
    ("My project uses {db} as its database.", "What database does the user's project use?", "{db}"),
    ("The API runs on port {port}.", "What port does the user's API run on?", "{port}"),
    ("We deploy to {cloud}.", "What cloud provider does the user deploy to?", "{cloud}"),
    # temporal
    ("The project deadline is {month2} {day2}.", "When is the user's project deadline?", "{month2} {day2}"),
    ("We launched version {ver} last week.", "What version was launched last week?", "{ver}"),
    ("The sprint ends on {weekday}.", "What day does the sprint end?", "{weekday}"),
    ("Our next release is in {nmonths} months.", "How many months until the next release?", "{nmonths}"),
    ("The incident happened at {hour}:00.", "At what time did the incident occur?", "{hour}:00"),
    # preference
    ("I prefer {style} code style.", "What code style does the user prefer?", "{style}"),
    ("I always use {testing} for testing.", "What testing framework does the user use?", "{testing}"),
    ("I dislike {antipattern}.", "What does the user dislike?", "{antipattern}"),
]

_FILLERS: list[tuple[str, str]] = [
    ("user", "Can you explain how gradient descent works?"),
    ("assistant", "Gradient descent iteratively adjusts parameters by moving opposite to the gradient of the loss."),
    ("user", "What's the difference between TCP and UDP?"),
    ("assistant", "TCP guarantees delivery with handshaking; UDP is connectionless and faster but unreliable."),
    ("user", "How does a hash table handle collisions?"),
    ("assistant", "Common strategies are chaining (linked lists per bucket) and open addressing (probing)."),
    ("user", "Explain the CAP theorem briefly."),
    ("assistant", "A distributed system can guarantee at most two of Consistency, Availability, Partition tolerance."),
    ("user", "What is memoization?"),
    ("assistant", "Memoization caches function results keyed by input so repeated calls skip recomputation."),
    ("user", "How do you reverse a linked list in-place?"),
    ("assistant", "Walk the list, rewiring next pointers backward using three pointers: prev, curr, next."),
    ("user", "What is a closure?"),
    ("assistant", "A closure is a function that captures bindings from its enclosing lexical scope."),
    ("user", "Explain Docker layers."),
    ("assistant", "Each Dockerfile instruction creates an immutable layer; layers are cached and shared across images."),
    ("user", "What is eventual consistency?"),
    ("assistant", "Replicas may diverge temporarily but converge to the same state given no new writes."),
    ("user", "How does LRU cache eviction work?"),
    ("assistant", "The least-recently-used entry is evicted when capacity is exceeded, via a hashmap + linked list."),
]

_NAMES = ["Alice Chen", "Bob Nakamura", "Clara Osei", "David Kim", "Eva Rossi"]
_CITIES = ["Berlin", "Seoul", "Lagos", "Toronto", "São Paulo"]
_COUNTRIES = ["Germany", "South Korea", "Nigeria", "Canada", "Brazil"]
_COMPANIES = ["Acme Corp", "NovaTech", "PixelWave", "DeepCore", "SkyLogic"]
_ROLES = ["senior engineer", "staff engineer", "principal architect", "lead developer", "data scientist"]
_MANAGERS = ["Sarah Jones", "Marcus Li", "Priya Gupta", "Tom Okonkwo", "Lena Müller"]
_LANGS = ["Python", "Rust", "Go", "TypeScript", "Elixir"]
_EDITORS = ["Neovim", "VS Code", "Emacs", "JetBrains Rider", "Zed"]
_DBS = ["PostgreSQL", "MongoDB", "Cassandra", "DynamoDB", "CockroachDB"]
_CLOUDS = ["AWS", "GCP", "Azure", "Fly.io", "Hetzner"]
_STYLES = ["functional", "object-oriented", "data-oriented", "pragmatic", "strictly typed"]
_TESTING = ["pytest", "Jest", "QuickCheck", "Hypothesis", "Vitest"]
_ANTIPATTERNS = ["global mutable state", "premature optimization", "God objects", "magic strings", "tight coupling"]
_MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
_WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]


def _sample_substitutions(rng: random.Random) -> dict[str, str]:
    return {
        "name": rng.choice(_NAMES),
        "city": rng.choice(_CITIES),
        "country": rng.choice(_COUNTRIES),
        "month": rng.choice(_MONTHS),
        "month2": rng.choice(_MONTHS),
        "day": str(rng.randint(1, 28)),
        "day2": str(rng.randint(1, 28)),
        "n": str(rng.randint(0, 5)),
        "colour": rng.choice(["red", "blue", "green", "purple", "teal", "orange"]),
        "company": rng.choice(_COMPANIES),
        "company2": rng.choice(_COMPANIES),
        "role": rng.choice(_ROLES),
        "salary": str(rng.randint(80, 250)),
        "years": str(rng.randint(1, 12)),
        "manager": rng.choice(_MANAGERS),
        "size": str(rng.randint(3, 20)),
        "lang": rng.choice(_LANGS),
        "editor": rng.choice(_EDITORS),
        "db": rng.choice(_DBS),
        "port": str(rng.choice([3000, 4000, 5000, 8000, 8080, 9000])),
        "cloud": rng.choice(_CLOUDS),
        "ver": f"{rng.randint(1, 5)}.{rng.randint(0, 9)}.{rng.randint(0, 9)}",
        "weekday": rng.choice(_WEEKDAYS),
        "nmonths": str(rng.randint(1, 6)),
        "hour": str(rng.randint(0, 23)),
        "style": rng.choice(_STYLES),
        "testing": rng.choice(_TESTING),
        "antipattern": rng.choice(_ANTIPATTERNS),
    }


def _apply(template: str, subs: dict[str, str]) -> str:
    out = template
    for k, v in subs.items():
        out = out.replace("{" + k + "}", v)
    return out


def _infer_category(tmpl: str) -> FactCategory:
    t = tmpl.lower()
    if any(k in t for k in ["name", "city", "birthday", "sibling", "colour"]):
        return "personal"
    if any(k in t for k in ["work", "salary", "company", "manager", "team"]):
        return "professional"
    if any(k in t for k in ["language", "editor", "database", "port", "deploy"]):
        return "technical"
    if any(k in t for k in ["deadline", "launch", "sprint", "release", "incident"]):
        return "temporal"
    return "preference"


def generate_conversation(
    *,
    n_facts: int = 20,
    n_turns: int = 60,
    noise_between: int = 2,
    seed: int = 42,
) -> SyntheticConversation:
    """Build a seeded synthetic conversation: ``n_facts`` planted across the first
    part of ``n_turns``, the rest filler, then one recall probe per fact."""
    rng = random.Random(seed)
    subs = _sample_substitutions(rng)

    templates = rng.sample(_FACT_BANK, min(n_facts, len(_FACT_BANK)))
    facts: list[Fact] = [
        Fact(
            id=f"fact_{i:03d}",
            category=_infer_category(stmt),
            statement=_apply(stmt, subs),
            probe=_apply(probe, subs),
            answer=_apply(ans, subs),
        )
        for i, (stmt, probe, ans) in enumerate(templates)
    ]

    turns: list[ConversationTurn] = []
    filler_pool = list(_FILLERS) * 10
    rng.shuffle(filler_pool)
    filler_idx = 0

    # Distribute fact plantings across the first part of the conversation, bounded so
    # positions never fall outside the loop.
    plant_window = min(max(n_turns - noise_between - n_facts, 5), n_turns - 1)
    fact_turns = set(sorted(rng.sample(range(0, plant_window), min(len(facts), plant_window))))
    fact_iter = iter(facts)

    for turn_i in range(n_turns):
        if turn_i in fact_turns:
            fact = next(fact_iter, None)
            if fact is not None:
                preamble = rng.choice(
                    ["By the way, ", "Just so you know, ", "I should mention that ",
                     "For context, ", "Oh and ", ""]
                )
                turns.append(ConversationTurn(
                    role="user",
                    content=f"{preamble}{fact.statement} Anyway, can you help me with something?",
                    contains_fact_ids=[fact.id],
                ))
                fact.planted_at_turn = turn_i
                turns.append(ConversationTurn(
                    role="assistant",
                    content="Got it, I've noted that. What would you like help with?",
                ))
                continue
        if filler_idx < len(filler_pool):
            role, content = filler_pool[filler_idx]
            filler_idx += 1
        else:
            role, content = "user", "Can you give me another technical tip?"
        turns.append(ConversationTurn(role=role, content=content))

    return SyntheticConversation(turns=turns, facts=facts, probes=list(facts))
