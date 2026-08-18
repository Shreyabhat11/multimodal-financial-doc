from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter

from app.api.schemas import HealthResponse

router = APIRouter(tags=["health"])

APP_VERSION = "0.1.0"


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Liveness/readiness check. Deliberately does NOT verify DB connectivity or VLM
    reachability on every call — a health check hit by a load balancer every few
    seconds should be cheap and fast. A deeper "are our dependencies actually up"
    check belongs in a separate /health/deep endpoint in a real deployment, not here.
    """
    return HealthResponse(status="ok", version=APP_VERSION, timestamp=datetime.now(timezone.utc))
