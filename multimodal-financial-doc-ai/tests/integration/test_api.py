"""
Integration tests: the FastAPI app, via TestClient, against a real temp-file SQLite
database and fake VLM backends. No real network calls anywhere in this file.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def api_client(sqlite_db_url, fake_good_vlm, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", sqlite_db_url)

    from app.database.session import init_db

    init_db()

    from app.api.dependencies import get_document_service
    from app.main import create_app
    from app.services.document_service import DocumentService

    fake_service = DocumentService(vision_model=fake_good_vlm, ocr_fallback_model=None)
    app = create_app()
    app.dependency_overrides[get_document_service] = lambda: fake_service

    with TestClient(app) as client:
        yield client


class TestHealthEndpoint:
    def test_health_check_returns_ok(self, api_client):
        response = api_client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"


class TestDocumentUploadFlow:
    def test_upload_process_and_retrieve_result(self, api_client, synthetic_pdf_bytes):
        response = api_client.post(
            "/documents/upload", files={"file": ("statement.pdf", synthetic_pdf_bytes, "application/pdf")}
        )
        assert response.status_code == 202
        document_id = response.json()["document_id"]

        status_response = api_client.get(f"/documents/{document_id}/status")
        assert status_response.status_code == 200
        assert status_response.json()["status"] == "completed"

        result_response = api_client.get(f"/documents/{document_id}/result")
        assert result_response.status_code == 200
        result = result_response.json()
        assert result["account"]["account_number"] == "******6655"  # masked
        assert len(result["transactions"]) == 2

    def test_reprocess_endpoint(self, api_client, synthetic_pdf_bytes):
        upload_response = api_client.post(
            "/documents/upload", files={"file": ("statement.pdf", synthetic_pdf_bytes, "application/pdf")}
        )
        document_id = upload_response.json()["document_id"]

        reprocess_response = api_client.post(f"/documents/{document_id}/reprocess")
        assert reprocess_response.status_code == 202

        status_response = api_client.get(f"/documents/{document_id}/status")
        assert status_response.json()["status"] == "completed"

    def test_rejects_non_pdf_upload(self, api_client):
        response = api_client.post(
            "/documents/upload", files={"file": ("statement.txt", b"not a pdf", "text/plain")}
        )
        assert response.status_code == 422
        assert response.json()["error_code"] == "document_processing_error"


class TestErrorHandling:
    def test_unknown_document_returns_404_with_structured_error(self, api_client):
        response = api_client.get("/documents/does-not-exist/status")
        assert response.status_code == 404
        body = response.json()
        assert body["error_code"] == "document_not_found"
        assert "does-not-exist" in body["message"]

    def test_unknown_document_result_returns_404(self, api_client):
        response = api_client.get("/documents/does-not-exist/result")
        assert response.status_code == 404
