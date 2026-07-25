"""Key normalization — the ONE place text becomes a comparison key.

Contradiction lookups and the single-valued integrity guard compare ``*_key``
columns, so every producer (extraction, temporal, repository) must normalize
identically or a real duplicate looks novel. Keep this boring and deterministic.
"""

from __future__ import annotations

import re
import unicodedata

_WS = re.compile(r"\s+")


def normalize_key(text: str) -> str:
    """Lowercase, strip accents, collapse whitespace, drop surrounding punctuation."""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.casefold().strip()
    text = _WS.sub(" ", text)
    return text.strip(" \t\r\n.,;:!?\"'`")
