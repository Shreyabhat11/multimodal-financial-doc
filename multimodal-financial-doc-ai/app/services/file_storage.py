"""
Uploaded-file storage.

Why this exists: POST /documents/{id}/reprocess (brief, Section 14) needs to re-run
the pipeline against the ORIGINAL uploaded bytes — without persisting them somewhere,
reprocessing would require the caller to re-upload the same file, which defeats the
purpose of a dedicated reprocess endpoint. Files are stored under `data/raw/`
(already scaffolded in Phase 1's project structure) keyed by document_id, so
retrieval never depends on the original filename.

This is a local-filesystem implementation deliberately kept behind a narrow
interface (`save`, `load`, `delete`) — swapping to S3/GCS/Azure Blob for a real
multi-instance deployment (where local disk isn't shared across API replicas) means
changing this one module, not any caller.
"""

from __future__ import annotations

from pathlib import Path

from app.core.exceptions import DocumentNotFoundError

_STORAGE_ROOT = Path(__file__).resolve().parents[2] / "data" / "raw"


def _path_for(document_id: str) -> Path:
    # document_id is always a server-generated UUID (never taken from user input for
    # path construction), so there's no path-traversal risk here — still worth
    # stating explicitly rather than leaving it as an unstated assumption.
    return _STORAGE_ROOT / f"{document_id}.pdf"


def save_uploaded_file(document_id: str, content: bytes) -> Path:
    _STORAGE_ROOT.mkdir(parents=True, exist_ok=True)
    path = _path_for(document_id)
    path.write_bytes(content)
    return path


def load_uploaded_file(document_id: str) -> bytes:
    path = _path_for(document_id)
    if not path.exists():
        raise DocumentNotFoundError(
            f"No stored file found for document '{document_id}' — it may have been "
            "deleted, or the document_id is invalid."
        )
    return path.read_bytes()


def delete_uploaded_file(document_id: str) -> None:
    path = _path_for(document_id)
    if path.exists():
        path.unlink()
