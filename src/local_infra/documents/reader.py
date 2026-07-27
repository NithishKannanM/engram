"""Bytes → plain text, with page offsets where the format has pages.

Every function here is **blocking** (pypdf and python-docx are synchronous, and a
40-page PDF takes seconds); the pipeline calls them inside ``asyncio.to_thread``. Kept
apart from the splitter so the offset arithmetic stays pure and testable while the
format-specific parsing — the part that needs real files — is isolated here.

Imports are deferred into each reader: pypdf and python-docx (via lxml) together cost
real import time and memory, and a service that only ever ingests Markdown should not
pay for either.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field

from local_infra.documents.validate import DocumentKind, ValidationError


@dataclass(frozen=True)
class ExtractedText:
    """Document text plus, for paginated formats, each page's start offset in ``text``.

    ``page_starts`` is empty for TXT/MD/DOCX — a DOCX has no page structure until it is
    laid out by a renderer, and inventing one would put a wrong page number on a
    citation, which is worse than showing none.
    """

    text: str
    page_starts: list[int] = field(default_factory=list)

    @property
    def pages(self) -> int | None:
        return len(self.page_starts) or None


#: Pages are joined with a blank line so the splitter's paragraph rule treats a page
#: break as a real boundary rather than gluing the last line of one page to the first
#: line of the next.
_PAGE_SEPARATOR = "\n\n"


def _read_pdf(data: bytes) -> ExtractedText:
    from pypdf import PdfReader
    from pypdf.errors import PdfReadError

    try:
        reader = PdfReader(io.BytesIO(data))
    except PdfReadError as exc:
        raise ValidationError(f"unreadable PDF: {exc}", status_code=400) from exc

    if reader.is_encrypted:
        # An empty-password decrypt succeeds for the common "owner password only" case;
        # anything else genuinely needs a secret we do not have.
        try:
            if not reader.decrypt(""):
                raise ValidationError("PDF is password-protected", status_code=400)
        except (NotImplementedError, PdfReadError) as exc:
            raise ValidationError(
                f"PDF uses unsupported encryption: {exc}", status_code=400
            ) from exc

    parts: list[str] = []
    page_starts: list[int] = []
    cursor = 0
    for page in reader.pages:
        try:
            body = page.extract_text() or ""
        except Exception:  # noqa: BLE001 — one malformed page must not lose the document
            body = ""
        page_starts.append(cursor)
        parts.append(body)
        cursor += len(body) + len(_PAGE_SEPARATOR)
    return ExtractedText(text=_PAGE_SEPARATOR.join(parts), page_starts=page_starts)


def _read_docx(data: bytes) -> ExtractedText:
    import docx

    try:
        document = docx.Document(io.BytesIO(data))
    except Exception as exc:  # noqa: BLE001 — python-docx raises several unrelated types
        raise ValidationError(f"unreadable .docx: {exc}", status_code=400) from exc

    blocks = [p.text for p in document.paragraphs if p.text.strip()]
    # Tables hold a lot of the actual facts in real documents and are invisible to
    # `paragraphs`. Each row becomes one line so a row stays one retrievable unit.
    for table in document.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells]
            line = " | ".join(c for c in cells if c)
            if line:
                blocks.append(line)
    return ExtractedText(text="\n\n".join(blocks))


def _read_text(data: bytes) -> ExtractedText:
    # `validate` already proved this decodes as UTF-8; errors="replace" is belt-and-braces
    # for the path where a caller skipped validation.
    return ExtractedText(text=data.decode("utf-8", errors="replace"))


def extract_text(data: bytes, kind: DocumentKind) -> ExtractedText:
    """Dispatch on the *validated* kind — never on the client's Content-Type."""
    if kind == "pdf":
        return _read_pdf(data)
    if kind == "docx":
        return _read_docx(data)
    if kind in ("txt", "md"):
        return _read_text(data)
    raise ValidationError(f"no reader for kind {kind!r}")
