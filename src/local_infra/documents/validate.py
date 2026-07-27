"""Upload validation. Nothing the client says about a file is trusted.

An upload arrives with three claims — filename, extension, and Content-Type — and every
one of them is attacker-controlled. So the *content* decides: magic bytes pick the kind,
an allowlist decides whether that kind is accepted, and the declared extension is used
only to disambiguate formats magic bytes cannot tell apart (TXT vs MD are both plain
text).

Pure and I/O-free: everything here takes ``bytes`` and returns a verdict, so the whole
threat model is unit-testable without a filesystem, a database, or an HTTP request.
"""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from typing import Literal

DocumentKind = Literal["pdf", "docx", "txt", "md"]

#: Streaming cap. Enforced *while reading*, never after — a 25MB limit checked on a
#: fully-buffered body is not a limit, it is a memory-exhaustion primitive.
MAX_BYTES = 25 * 1024 * 1024

#: Enough for every signature below, including the tar magic at offset 257.
SNIFF_BYTES = 512

#: Uncompressed:compressed ratio above which a zip is treated as a bomb. Real DOCX files
#: land around 3–15x (XML compresses well); 42.zip is ~1e9x.
MAX_ZIP_RATIO = 200
#: Absolute ceiling on a DOCX's inflated size, for the ratio-evading "many small files"
#: variant where each entry looks individually reasonable.
MAX_ZIP_TOTAL_BYTES = 200 * 1024 * 1024


class ValidationError(Exception):
    """Rejected upload. ``status_code`` is the HTTP status the route should return —
    415 (wrong type), 413 (too large), 400 (malformed). Carrying it here keeps the route
    a thin translation layer and keeps *why* next to the rule that decided it."""

    def __init__(self, message: str, *, status_code: int = 415) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True)
class ValidatedUpload:
    kind: DocumentKind
    filename: str
    size: int
    sha256: str


# --------------------------------------------------------------------------- #
# Signatures                                                                   #
# --------------------------------------------------------------------------- #
#: Executables and archives, rejected by name so the error says what was detected
#: instead of a generic "unsupported". Order matters only for readability; the checks
#: are disjoint.
_DENY_MAGIC: tuple[tuple[bytes, str], ...] = (
    (b"\x7fELF", "an ELF executable"),
    (b"MZ", "a Windows executable"),
    (b"\xfe\xed\xfa\xce", "a Mach-O executable"),
    (b"\xfe\xed\xfa\xcf", "a Mach-O executable"),
    (b"\xce\xfa\xed\xfe", "a Mach-O executable"),
    (b"\xcf\xfa\xed\xfe", "a Mach-O executable"),
    (b"\xca\xfe\xba\xbe", "a Mach-O fat binary"),
    (b"#!", "a script"),
    (b"\x1f\x8b", "a gzip archive"),
    (b"BZh", "a bzip2 archive"),
    (b"\xfd7zXZ\x00", "an xz archive"),
    (b"\x28\xb5\x2f\xfd", "a zstd archive"),
    (b"7z\xbc\xaf\x27\x1c", "a 7-zip archive"),
    (b"Rar!\x1a\x07", "a RAR archive"),
    (b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1", "a legacy OLE2 document (.doc/.xls)"),
)

_ZIP_MAGIC = (b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08")
_PDF_MAGIC = b"%PDF-"

_TEXT_EXTENSIONS: dict[str, DocumentKind] = {
    "md": "md",
    "markdown": "md",
    "txt": "txt",
    "text": "txt",
    "": "txt",
}


def safe_filename(name: str | None) -> str:
    """Basename only, control characters stripped.

    The on-disk name is the sha256, so this string is display metadata — but it still
    reaches a UI and a JSON response, and ``../../.ssh/authorized_keys`` should never be
    echoed back as if it were a path we honoured.
    """
    raw = (name or "").replace("\\", "/").split("/")[-1]
    cleaned = "".join(ch for ch in raw if ch.isprintable() and ch != "\x00").strip()
    cleaned = cleaned.lstrip(".") or "document"
    return cleaned[:255]


def _extension(filename: str) -> str:
    base = safe_filename(filename)
    _, dot, ext = base.rpartition(".")
    return ext.lower() if dot else ""


def enforce_size(total: int) -> None:
    """Raise once a streamed read passes the cap. Call per chunk, not at the end."""
    if total > MAX_BYTES:
        raise ValidationError(
            f"file exceeds the {MAX_BYTES // (1024 * 1024)}MB limit", status_code=413
        )


def _looks_like_text(data: bytes) -> bool:
    """UTF-8-decodable and free of NUL bytes.

    The NUL check is the cheap discriminator between text and an unrecognised binary:
    no legitimate UTF-8 document contains one, and every binary format that slipped past
    the magic checks almost certainly does.
    """
    if b"\x00" in data:
        return False
    try:
        data.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return True


def _check_docx(data: bytes) -> None:
    """A DOCX is a zip, so the ``PK`` sniff alone accepts *any* zip — a jar, a zip bomb,
    an APK. Require the OOXML entry, reject traversal, and bound the inflated size."""
    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as exc:
        raise ValidationError(f"not a readable .docx: {exc}", status_code=400) from exc

    with zf:
        names = zf.namelist()
        if "word/document.xml" not in names:
            raise ValidationError(
                "zip archive is not a Word document (no word/document.xml)"
            )
        for name in names:
            normalized = name.replace("\\", "/")
            if normalized.startswith("/") or ".." in normalized.split("/"):
                raise ValidationError(
                    f"archive entry escapes its root: {name!r}", status_code=400
                )
        total = sum(info.file_size for info in zf.infolist())
        if total > MAX_ZIP_TOTAL_BYTES:
            raise ValidationError(
                "archive expands beyond the decompression limit", status_code=413
            )
        if total > len(data) * MAX_ZIP_RATIO:
            raise ValidationError(
                f"archive compression ratio {total // max(1, len(data))}:1 exceeds "
                f"{MAX_ZIP_RATIO}:1 (possible zip bomb)",
                status_code=413,
            )


def detect_kind(data: bytes, filename: str) -> DocumentKind:
    """Decide the kind from content, using the extension only where content cannot tell.

    Raises :class:`ValidationError` for anything outside the allowlist. This is an
    allowlist by construction: the final branch rejects, it does not default.
    """
    head = data[:SNIFF_BYTES]

    for magic, described in _DENY_MAGIC:
        if head.startswith(magic):
            raise ValidationError(f"refusing {described}")
    # tar keeps its magic at offset 257, so a prefix check cannot see it.
    if len(data) > 262 and data[257:262] == b"ustar":
        raise ValidationError("refusing a tar archive")

    if head.startswith(_PDF_MAGIC):
        return "pdf"

    if head.startswith(_ZIP_MAGIC):
        _check_docx(data)
        return "docx"

    if _looks_like_text(data):
        return _TEXT_EXTENSIONS.get(_extension(filename), "txt")

    raise ValidationError(
        "unsupported file type — accepted: PDF, DOCX, TXT, MD"
    )


def validate_upload(data: bytes, filename: str | None) -> tuple[DocumentKind, str]:
    """Full check on complete bytes. Returns ``(kind, safe_filename)``.

    Callers that stream must still call :func:`enforce_size` per chunk; this re-checks
    the total so a non-streaming caller cannot bypass the cap, and rejects an empty body
    (which would otherwise sniff as valid text and ingest as a zero-chunk document).
    """
    enforce_size(len(data))
    if not data:
        raise ValidationError("file is empty", status_code=400)
    name = safe_filename(filename)
    return detect_kind(data, name), name
