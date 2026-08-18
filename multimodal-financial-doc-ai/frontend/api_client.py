"""
Thin wrapper around the FastAPI backend's HTTP surface.

Kept deliberately dumb (no retries, no caching, no business logic) — the Streamlit
app is a client of the API, not a second implementation of the pipeline. Every
function here maps 1:1 to one API endpoint from app/api/routes/documents.py.
Isolating these calls in one module (rather than scattering `requests.post(...)`
calls through the UI code) is what makes the UI testable independent of a running
API server — tests can monkeypatch this module's functions.
"""

from __future__ import annotations

from dataclasses import dataclass

import requests


class ApiError(Exception):
    """Raised when the backend returns a non-2xx response. Carries the parsed
    error_code/message from the backend's structured ErrorResponse (Phase 12) when
    available, so the UI can show the same message a developer hitting the API
    directly would see, rather than a generic 'something went wrong'."""

    def __init__(self, status_code: int, error_code: str, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.error_code = error_code
        self.message = message


@dataclass
class ApiClient:
    base_url: str
    timeout_seconds: float = 30.0

    def _handle_response(self, response: requests.Response) -> dict:
        if response.ok:
            return response.json()
        try:
            body = response.json()
            raise ApiError(
                response.status_code,
                body.get("error_code", "unknown_error"),
                body.get("message", response.text),
            )
        except ValueError:  # response body wasn't JSON
            raise ApiError(response.status_code, "unknown_error", response.text) from None

    def health_check(self) -> dict:
        response = requests.get(f"{self.base_url}/health", timeout=self.timeout_seconds)
        return self._handle_response(response)

    def upload_document(self, *, file_bytes: bytes, filename: str) -> dict:
        response = requests.post(
            f"{self.base_url}/documents/upload",
            files={"file": (filename, file_bytes, "application/pdf")},
            timeout=self.timeout_seconds,
        )
        return self._handle_response(response)

    def get_status(self, document_id: str) -> dict:
        response = requests.get(f"{self.base_url}/documents/{document_id}/status", timeout=self.timeout_seconds)
        return self._handle_response(response)

    def get_result(self, document_id: str) -> dict:
        response = requests.get(f"{self.base_url}/documents/{document_id}/result", timeout=self.timeout_seconds)
        return self._handle_response(response)

    def reprocess_document(self, document_id: str) -> dict:
        response = requests.post(f"{self.base_url}/documents/{document_id}/reprocess", timeout=self.timeout_seconds)
        return self._handle_response(response)
