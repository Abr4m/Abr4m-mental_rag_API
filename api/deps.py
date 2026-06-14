"""
API: Dependencies
FastAPI dependency-injection functions that resolve services from the container.
Previously at api/dependencies/injection.py.
"""
from fastapi import Depends

from app.database.container import AppContainer, get_container
from app.services.query_service import QueryMentalHealthSupportService
from app.services.tags_service import GetAvailableTagsService


def get_query_service(
    container: AppContainer = Depends(get_container),
) -> QueryMentalHealthSupportService:
    return QueryMentalHealthSupportService(
        repository=container.repository,
        retriever=container.retriever,
        reranker=container.reranker,
        llm_processor=container.llm_processor,
    )


def get_tags_service(
    container: AppContainer = Depends(get_container),
) -> GetAvailableTagsService:
    return GetAvailableTagsService(repository=container.repository)
