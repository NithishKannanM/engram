"""Content-addressed blob storage on the local filesystem.

Bytes live here, not in Postgres: a 25MB PDF in a column is pulled into memory by every
metadata query that forgets to name its columns, and it bloats every dump. The path is
``<root>/<sha[:2]>/<sha>`` — the two-character shard keeps a single directory from
growing to tens of thousands of entries, which some filesystems handle badly.

Content-addressed means writes are naturally idempotent and the same bytes uploaded into
two scopes are stored once. That is also why deletion is *reference-counted by the
database*, never by the caller who happens to be deleting a document: see
`DocumentStore.sha_in_use_elsewhere`.

Synchronous by design (this is filesystem I/O); the pipeline calls it via
``asyncio.to_thread``.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path


def sha256_of(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class BlobStore:
    def __init__(self, root: Path) -> None:
        self._root = Path(root)

    @property
    def root(self) -> Path:
        return self._root

    def path_for(self, sha: str) -> Path:
        # `sha` is always produced by `sha256_of`, never by a client — but slicing a
        # hex digest keeps it path-safe by construction regardless.
        return self._root / sha[:2] / sha

    def exists(self, sha: str) -> bool:
        return self.path_for(sha).is_file()

    def put(self, data: bytes) -> str:
        """Store ``data``, returning its sha256. Idempotent.

        Written to a temp file in the destination directory and then ``os.replace``d,
        which is atomic on POSIX: a crash mid-write can leave a stray temp file but
        never a truncated blob that would later be read back as a corrupt document.
        """
        sha = sha256_of(data)
        target = self.path_for(sha)
        if target.is_file():
            return sha
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=target.parent, prefix=".tmp-")
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(data)
            os.replace(tmp, target)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
        return sha

    def read(self, sha: str) -> bytes:
        return self.path_for(sha).read_bytes()

    def unlink(self, sha: str) -> bool:
        """Remove a blob. Returns False when it was already gone — a delete racing
        another delete is not an error worth failing a request over."""
        path = self.path_for(sha)
        try:
            path.unlink()
        except FileNotFoundError:
            return False
        # Prune the shard directory when it empties; ignore it when it does not.
        try:
            path.parent.rmdir()
        except OSError:
            pass
        return True
