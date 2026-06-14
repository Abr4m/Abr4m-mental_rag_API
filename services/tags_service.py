"""
Services: GetAvailableTagsService
Returns the sorted list of tags the client can use as filters.
Previously at application/use_cases/get_tags.py.
"""
from typing import List

from app.models.interfaces import SuggestionRepository


class GetAvailableTagsService:
    def __init__(self, repository: SuggestionRepository) -> None:
        self._repo = repository

    def execute(self) -> List[str]:
        return self._repo.get_available_tags()
