"""
Router: Query
POST /query — run the full RAG pipeline and return an LLM-synthesised answer.
"""
import logging

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.deps import get_query_service
from app.schemas.query import QueryRequest, QueryResponse, SourceCardResponse
from app.services.query_service import QueryMentalHealthSupportService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/query", tags=["Query"])


@router.post(
    "",
    response_model=QueryResponse,
    summary="Mental health support query",
    description=(
        "Submit a free-text query. The API retrieves community suggestions via "
        "bi-encoder + FAISS, re-ranks with a cross-encoder, and synthesises a "
        "compassionate response using Qwen2.5-1.5B-Instruct."
    ),
)
def query(
    request: QueryRequest,
    service: QueryMentalHealthSupportService = Depends(get_query_service),
) -> QueryResponse:
    try:
        result = service.execute(
            query=request.query,
            tag_filter=request.tag_filter,
            n_tips=request.n_tips,
        )
    except Exception as exc:
        logger.exception("Unhandled error in query service: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal error occurred while processing your query.",
        ) from exc

    sources = [
        SourceCardResponse(
            concern=s.concern,
            title=s.title,
            tag=s.tag,
            all_labels=s.all_labels,
            suggestions=s.suggestions,
        )
        for s in result.sources
    ]
    return QueryResponse(answer=result.answer, sources=sources)
