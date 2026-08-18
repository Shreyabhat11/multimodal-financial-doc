"""
FastAPI application entrypoint.

`create_app()` is a factory (not a bare module-level `app = FastAPI()`) so that tests
can construct fresh app instances with different settings/overrides without import-
order side effects — a common FastAPI testing pitfall when the app object is built
once at import time with no way to reconfigure it per test.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.exception_handlers import register_exception_handlers
from app.api.routes import documents, health
from app.core.config import get_settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="Multimodal Financial Document Understanding API",
        description=(
            "Upload financial documents (bank statements, credit card statements, invoices, "
            "loan statements) for multimodal VLM-based extraction, deterministic financial "
            "validation, and CrewAI-based review."
        ),
        version="0.1.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api_cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    register_exception_handlers(app)

    app.include_router(health.router)
    app.include_router(documents.router)

    return app


app = create_app()
