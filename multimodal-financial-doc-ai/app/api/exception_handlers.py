"""
Exception handlers.

Every custom exception in app.core.exceptions carries its own `http_status` and
`error_code` — registering ONE handler for the base `AppError` class here is enough
to correctly handle every subclass (FileTooLargeError, DocumentNotFoundError,
CrewValidationError, etc.) without a handler per exception type. This is exactly why
Phase 2 built a real exception hierarchy instead of raising bare `ValueError`/
`Exception` throughout — it pays off concretely here.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from app.api.schemas import ErrorResponse
from app.core.exceptions import AppError

logger = logging.getLogger("app.api")


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def handle_app_error(request: Request, exc: AppError) -> JSONResponse:
        logger.warning("AppError on %s %s: %s", request.method, request.url.path, exc.to_dict())
        return JSONResponse(
            status_code=exc.http_status,
            content=ErrorResponse(error_code=exc.error_code, message=exc.message, details=exc.details).model_dump(),
        )

    @app.exception_handler(RequestValidationError)
    async def handle_request_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content=ErrorResponse(
                error_code="request_validation_error",
                message="The request did not match the expected shape.",
                details={"errors": exc.errors()},
            ).model_dump(),
        )

    @app.exception_handler(ValidationError)
    async def handle_pydantic_validation_error(request: Request, exc: ValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content=ErrorResponse(
                error_code="validation_error",
                message="A data validation error occurred.",
                details={"errors": exc.errors()},
            ).model_dump(),
        )

    @app.exception_handler(Exception)
    async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        # Deliberately generic message to the client (never leak internal exception
        # text/stack details over the API) — the real detail goes to the server log
        # only. This is the catch-all safety net for genuinely unanticipated bugs;
        # every EXPECTED failure mode should already be caught upstream and raised as
        # a specific AppError subclass, so reaching this handler is itself a signal
        # worth investigating, not routine.
        logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error_code="internal_server_error",
                message="An unexpected error occurred. Please try again or contact support.",
            ).model_dump(),
        )
