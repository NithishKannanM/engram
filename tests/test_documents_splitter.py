"""Splitter unit tests. Pure — no DB, no embedder, no network."""

from __future__ import annotations

from local_infra.documents.splitter import (
    CHARS_PER_TOKEN,
    TextChunk,
    estimate_tokens,
    page_of,
    split_text,
)


def _paragraph(word: str, n: int) -> str:
    return " ".join([word] * n)


def test_empty_and_whitespace_produce_no_chunks():
    assert split_text("") == []
    assert split_text("   \n\n \t ") == []


def test_short_document_is_one_chunk_verbatim():
    text = "Priya works at Globex. She joined in March."
    (chunk,) = split_text(text)
    assert chunk.text == text
    assert (chunk.char_start, chunk.char_end, chunk.core_start) == (0, len(text), 0)
    assert chunk.index == 0


def test_every_chunk_is_an_exact_slice_of_the_source():
    """The invariant the citation drawer depends on: a chunk is a *span* of the original,
    not a re-joined approximation, so highlighting `[char_start:char_end]` is exact."""
    text = "\n\n".join(_paragraph(f"para{i}", 60) for i in range(12))
    chunks = split_text(text)
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.text == text[chunk.char_start : chunk.char_end]
        assert chunk.char_start <= chunk.core_start < chunk.char_end


def test_chunks_cover_every_non_whitespace_character():
    """Overlap may repeat text; nothing may be dropped. A silent gap is unfindable —
    the fact is simply never retrievable and no error is ever raised."""
    text = "\n\n".join(_paragraph(f"w{i}", 50) for i in range(20))
    chunks = split_text(text)
    assert chunks[0].char_start == 0 and chunks[-1].char_end == len(text)
    for previous, nxt in zip(chunks, chunks[1:]):
        gap = text[previous.char_end : nxt.char_start]
        assert gap.strip() == "", f"dropped source text: {gap!r}"


def test_overlap_duplicates_the_tail_of_the_previous_chunk():
    text = "\n\n".join(_paragraph(f"w{i}", 60) for i in range(10))
    chunks = split_text(text, max_tokens=120, overlap_tokens=20)
    assert len(chunks) > 2
    second = chunks[1]
    assert second.overlap_chars > 0
    # The duplicated lead-in is the previous chunk's tail (modulo the separator
    # whitespace the slice necessarily carries, since a chunk is a contiguous span).
    lead_in = text[second.char_start : second.core_start].strip()
    assert lead_in and lead_in in chunks[0].text


def test_overlap_can_be_disabled():
    text = "\n\n".join(_paragraph(f"w{i}", 60) for i in range(6))
    chunks = split_text(text, max_tokens=100, min_tokens=0, overlap_tokens=0)
    assert all(c.overlap_chars == 0 for c in chunks)
    # With no overlap the chunks tile the document: same content, nothing repeated.
    squash = lambda s: "".join(s.split())  # noqa: E731
    assert squash("".join(c.text for c in chunks)) == squash(text)


def test_paragraphs_are_preferred_boundaries():
    """Two paragraphs that both fit stay whole rather than being cut mid-sentence."""
    text = f"{_paragraph('alpha', 40)}\n\n{_paragraph('beta', 40)}"
    chunks = split_text(text, max_tokens=60, min_tokens=0, overlap_tokens=0)
    assert len(chunks) == 2
    assert chunks[0].text.split()[0] == "alpha"
    assert chunks[1].text.split()[0] == "beta"


def test_long_paragraph_falls_back_to_sentences():
    sentences = " ".join(f"Sentence number {i} says something useful." for i in range(60))
    chunks = split_text(sentences, max_tokens=60, overlap_tokens=0)
    assert len(chunks) > 1
    # Sentence granularity: no chunk starts mid-word.
    for chunk in chunks:
        assert chunk.text[0].isalnum()


def test_unbroken_run_is_hard_split_and_still_bounded():
    """A minified JSON blob or a one-line table has no paragraph or sentence structure.
    Without a hard split it would become a single chunk far past the embedder's window."""
    text = "x" * 5000
    chunks = split_text(text, max_tokens=50, min_tokens=0, overlap_tokens=0)
    assert len(chunks) > 1
    assert all(c.tokens <= 50 for c in chunks)


def test_hard_split_prefers_word_boundaries():
    text = " ".join("word" for _ in range(400))
    chunks = split_text(text, max_tokens=40, min_tokens=0, overlap_tokens=0)
    for chunk in chunks:
        assert chunk.text.startswith("word") and chunk.text.endswith("word")


def test_runt_tail_is_folded_into_the_previous_chunk():
    """A 3-word trailing chunk embeds to noise and then matches everything."""
    body = "\n\n".join(_paragraph(f"w{i}", 40) for i in range(6))
    text = f"{body}\n\ntiny tail"
    chunks = split_text(text, max_tokens=31, min_tokens=20, overlap_tokens=0)
    assert chunks[-1].text.endswith("tiny tail")
    assert len(chunks[-1].text) > len("tiny tail")


def test_min_tokens_zero_keeps_the_tail_separate():
    body = "\n\n".join(_paragraph(f"w{i}", 40) for i in range(6))
    text = f"{body}\n\ntiny tail"
    chunks = split_text(text, max_tokens=31, min_tokens=0, overlap_tokens=0)
    assert chunks[-1].text == "tiny tail"


def test_min_tokens_above_max_tokens_does_not_collapse_the_document():
    """A misconfiguration that would otherwise fold every chunk into the first one and
    return a single document-sized 'chunk' with no error anywhere."""
    text = "\n\n".join(_paragraph(f"w{i}", 40) for i in range(8))
    chunks = split_text(text, max_tokens=40, min_tokens=500, overlap_tokens=0)
    assert len(chunks) > 1


def test_chunks_are_indexed_in_order():
    text = "\n\n".join(_paragraph(f"w{i}", 50) for i in range(10))
    chunks = split_text(text, max_tokens=80)
    assert [c.index for c in chunks] == list(range(len(chunks)))
    assert all(a.char_start < b.char_start for a, b in zip(chunks, chunks[1:]))


def test_estimate_tokens_is_the_documented_heuristic():
    assert estimate_tokens("") == 0
    assert estimate_tokens("a" * CHARS_PER_TOKEN) == 1
    assert estimate_tokens("a" * (CHARS_PER_TOKEN * 3 + 1)) == 4


def test_chunk_is_frozen():
    (chunk,) = split_text("one small document")
    assert isinstance(chunk, TextChunk)


# --- page mapping ---------------------------------------------------------- #


def test_page_of_maps_offsets_to_one_based_pages():
    starts = [0, 100, 250]
    assert page_of(starts, 0) == 1
    assert page_of(starts, 99) == 1
    assert page_of(starts, 100) == 2
    assert page_of(starts, 249) == 2
    assert page_of(starts, 250) == 3
    assert page_of(starts, 10_000) == 3


def test_page_of_without_pages_is_none():
    """TXT/MD/DOCX have no pages; a fabricated page number on a citation is worse than
    no page number at all."""
    assert page_of([], 42) is None


def test_importing_the_document_stack_does_not_import_litellm():
    """Same guarantee the chat layer is pinned to, extended to documents.

    `import litellm` fetches its model-cost map over the NETWORK at import time and
    costs ~2.2s / 213MB. RAG touches `llm/prompt.py` and the pipeline touches
    `llm/gpu.py`, so it would be easy for the document stack to drag it in — and
    `test_offline_guard.py` cannot catch it, since its socket patch runs long after
    imports happen at collection.
    """
    import sys

    for module in (
        "local_infra.documents.pipeline",
        "local_infra.documents.rag",
        "local_infra.documents.service",
        "local_infra.api.routes_documents",
    ):
        assert module in sys.modules or __import__(module)
    assert "litellm" not in sys.modules
