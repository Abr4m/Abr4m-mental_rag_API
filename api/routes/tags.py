"""
Router: Tags
GET /tags — returns all valid tag values for the query filter.
"""
from fastapi import APIRouter, Depends

from app.api.deps import get_tags_service
from app.schemas.query import TagsResponse
from app.services.tags_service import GetAvailableTagsService

router = APIRouter(prefix="/tags", tags=["Tags"])


@router.get("", response_model=TagsResponse, summary="List available tag filters")
def list_tags(
    service: GetAvailableTagsService = Depends(get_tags_service),
) -> TagsResponse:
    return TagsResponse(tags=service.execute())
