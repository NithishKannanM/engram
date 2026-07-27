"""Upload validation tests — the threat model, asserted.

Everything here is pure bytes in, verdict out: no filesystem, no HTTP, no database.
"""

from __future__ import annotations

import io
import zipfile

import pytest

from local_infra.documents.reader import extract_text
from local_infra.documents.splitter import page_of
from local_infra.documents.validate import (
    MAX_BYTES,
    ValidationError,
    detect_kind,
    enforce_size,
    safe_filename,
    validate_upload,
)


def _zip(entries: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, body in entries.items():
            zf.writestr(name, body)
    return buf.getvalue()


def _docx_bytes(text: str = "Priya works at Globex.") -> bytes:
    import docx

    document = docx.Document()
    document.add_paragraph(text)
    buf = io.BytesIO()
    document.save(buf)
    return buf.getvalue()


# --- accepted kinds -------------------------------------------------------- #


def test_plain_text_is_accepted_and_extension_picks_txt_vs_md():
    assert detect_kind(b"hello world", "notes.txt") == "txt"
    assert detect_kind(b"# Heading", "notes.md") == "md"
    assert detect_kind(b"# Heading", "notes.markdown") == "md"
    # No extension at all still ingests — as plain text, not as a rejection.
    assert detect_kind(b"hello", "notes") == "txt"


def test_pdf_is_detected_by_magic_not_by_extension():
    """A PDF named .txt is still a PDF; a text file named .pdf is still text. Content
    decides, because the filename is attacker-controlled."""
    assert detect_kind(b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\n", "anything.txt") == "pdf"
    assert detect_kind(b"just words", "lies.pdf") == "txt"


def test_real_docx_is_accepted():
    assert detect_kind(_docx_bytes(), "report.docx") == "docx"


# --- rejected content ------------------------------------------------------ #


@pytest.mark.parametrize(
    ("payload", "described"),
    [
        (b"\x7fELF\x02\x01\x01" + b"\x00" * 32, "ELF executable"),
        (b"MZ\x90\x00\x03" + b"\x00" * 32, "Windows executable"),
        (b"#!/bin/sh\nrm -rf /\n", "script"),
        (b"\x1f\x8b\x08\x00" + b"\x00" * 32, "gzip archive"),
        (b"7z\xbc\xaf\x27\x1c" + b"\x00" * 32, "7-zip archive"),
        (b"Rar!\x1a\x07\x00" + b"\x00" * 32, "RAR archive"),
        (b"\x28\xb5\x2f\xfd" + b"\x00" * 32, "zstd archive"),
        (b"\xcf\xfa\xed\xfe" + b"\x00" * 32, "Mach-O executable"),
        (b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\x00" * 32, "legacy OLE2"),
    ],
)
def test_executables_and_archives_are_refused(payload: bytes, described: str):
    with pytest.raises(ValidationError) as exc:
        detect_kind(payload, "payload.txt")
    assert exc.value.status_code == 415


def test_tar_is_refused_despite_its_magic_living_at_offset_257():
    tar = bytearray(b"\x00" * 1024)
    tar[0:8] = b"file.txt"
    tar[257:262] = b"ustar"
    with pytest.raises(ValidationError, match="tar"):
        detect_kind(bytes(tar), "archive.tar")


def test_unrecognised_binary_is_refused_by_the_nul_check():
    """Nothing above matched, and it is not decodable text — the allowlist rejects
    rather than defaulting to 'probably fine'."""
    with pytest.raises(ValidationError, match="unsupported file type"):
        detect_kind(b"\x89rand\x00om\xffbytes", "mystery.txt")


def test_invalid_utf8_is_refused():
    with pytest.raises(ValidationError, match="unsupported file type"):
        detect_kind(b"\xc3\x28 broken continuation", "notes.txt")


# --- the zip family: PK alone is not enough -------------------------------- #


def test_a_zip_that_is_not_a_word_document_is_refused():
    """The `PK` sniff accepts every zip — a jar, an APK, a backup. Requiring
    word/document.xml is what makes 'DOCX' mean DOCX."""
    payload = _zip({"META-INF/MANIFEST.MF": b"Manifest-Version: 1.0\n"})
    with pytest.raises(ValidationError, match="not a Word document"):
        detect_kind(payload, "app.docx")


def test_zip_entry_escaping_its_root_is_refused():
    payload = _zip(
        {"word/document.xml": b"<w:document/>", "../../etc/passwd": b"root:x:0:0"}
    )
    with pytest.raises(ValidationError, match="escapes its root") as exc:
        detect_kind(payload, "evil.docx")
    assert exc.value.status_code == 400


def test_zip_bomb_is_refused_by_compression_ratio():
    payload = _zip({"word/document.xml": b"<w:document/>", "bomb.bin": b"\x00" * 40_000_000})
    with pytest.raises(ValidationError, match="compression ratio") as exc:
        detect_kind(payload, "bomb.docx")
    assert exc.value.status_code == 413


def test_corrupt_zip_is_a_400_not_a_500():
    with pytest.raises(ValidationError) as exc:
        detect_kind(b"PK\x03\x04" + b"\x00" * 64, "truncated.docx")
    assert exc.value.status_code == 400


# --- size + filename ------------------------------------------------------- #


def test_enforce_size_raises_413_past_the_cap():
    enforce_size(MAX_BYTES)  # exactly at the cap is fine
    with pytest.raises(ValidationError) as exc:
        enforce_size(MAX_BYTES + 1)
    assert exc.value.status_code == 413


def test_empty_upload_is_rejected():
    """Empty bytes sniff as valid text and would ingest as a zero-chunk document that
    silently returns nothing forever."""
    with pytest.raises(ValidationError) as exc:
        validate_upload(b"", "empty.txt")
    assert exc.value.status_code == 400


def test_safe_filename_strips_paths_and_control_characters():
    assert safe_filename("../../etc/passwd") == "passwd"
    assert safe_filename(r"C:\Users\me\report.pdf") == "report.pdf"
    assert safe_filename("nul\x00byte.txt") == "nulbyte.txt"
    assert safe_filename(".hidden") == "hidden"
    assert safe_filename(None) == "document"
    assert safe_filename("   ") == "document"
    assert len(safe_filename("a" * 400 + ".txt")) <= 255


def test_validate_upload_returns_kind_and_clean_name():
    kind, name = validate_upload(b"# Notes", "../../notes.md")
    assert (kind, name) == ("md", "notes.md")


# --- reader ---------------------------------------------------------------- #


def test_reader_extracts_docx_paragraphs_and_tables():
    import docx

    document = docx.Document()
    document.add_paragraph("Priya works at Globex.")
    table = document.add_table(rows=1, cols=2)
    table.rows[0].cells[0].text = "Role"
    table.rows[0].cells[1].text = "Staff Engineer"
    buf = io.BytesIO()
    document.save(buf)

    extracted = extract_text(buf.getvalue(), "docx")
    assert "Priya works at Globex." in extracted.text
    # Table rows carry real facts and are invisible to `document.paragraphs`.
    assert "Role | Staff Engineer" in extracted.text
    # A DOCX has no pages until a renderer lays it out.
    assert extracted.page_starts == [] and extracted.pages is None


def test_reader_decodes_text_without_pages():
    extracted = extract_text("héllo — unicode".encode(), "txt")
    assert extracted.text == "héllo — unicode"
    assert extracted.pages is None


def test_reader_extracts_pdf_text_with_page_offsets():
    from tests.pdf_fixture import build_pdf

    data = build_pdf(
        [
            ["Priya works at Globex.", "She joined in March 2024."],
            ["Marcus manages the platform team."],
        ]
    )
    assert detect_kind(data, "notes.pdf") == "pdf"

    extracted = extract_text(data, "pdf")
    assert "Priya works at Globex." in extracted.text
    assert "Marcus manages the platform team." in extracted.text
    assert extracted.pages == 2

    # The page offsets must index into the SAME string the splitter sees, or every
    # citation lands on the wrong page.
    assert page_of(extracted.page_starts, extracted.text.index("Priya")) == 1
    assert page_of(extracted.page_starts, extracted.text.index("Marcus")) == 2


def test_reader_rejects_a_truncated_pdf_as_400():
    """Malformed input is the client's error, not a server crash."""
    with pytest.raises(ValidationError) as exc:
        extract_text(b"%PDF-1.4\ngarbage, no xref, no pages", "pdf")
    assert exc.value.status_code == 400
