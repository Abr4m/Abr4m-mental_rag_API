"""
Router: Health
GET /health — liveness check for Lightning AI health probes.
"""
from fastapi import APIRouter

from app.core.config import get_settings
from app.schemas.query import HealthResponse

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse, summary="Liveness check")
def health() -> HealthResponse:
    settings = get_settings()
    return HealthResponse(status="ok", version=settings.api_version)
