"""Cloud spend ceiling.

Cloud is the only rung that costs money; this caps it per process. A zero ceiling
(the default) means the ladder is effectively local-only even when not strictly
offline — cloud rungs defer instead of spending.
"""

from __future__ import annotations


class Budget:
    def __init__(self, ceiling_usd: float = 0.0) -> None:
        self._ceiling = ceiling_usd
        self._spent = 0.0

    @property
    def remaining(self) -> float:
        return max(0.0, self._ceiling - self._spent)

    def can_spend(self, estimate: float) -> bool:
        return self._spent + estimate <= self._ceiling

    def record(self, amount: float) -> None:
        self._spent += amount
