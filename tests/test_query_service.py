"""
Tests: QueryMentalHealthSupportService Unit Tests
All infrastructure is mocked — no models are loaded.
"""
import pytest

from app.services.query_service import QueryMentalHealthSupportService
from app.tests.conftest import StubLLM, StubReranker, StubRepository, StubRetriever


@pytest.fixture
def service():
    return QueryMentalHealthSupportService(
        repository=StubRepository(),
        retriever=StubRetriever(),
        reranker=StubReranker(),
        llm_processor=StubLLM(),
    )


def test_returns_answer(service):
    result = service.execute("I feel anxious")
    assert result.answer == "Stub answer."


def test_returns_sources(service):
    result = service.execute("I feel anxious")
    assert len(result.sources) > 0
    assert result.sources[0].concern in ("Concern A", "Concern B")


def test_tag_filter_no_match(service):
    result = service.execute("I feel anxious", tag_filter="nonexistent_tag")
    assert "No results" in result.answer


def test_n_tips_limit(service):
    result = service.execute("I feel anxious", n_tips=1)
    total_suggestions = sum(len(s.suggestions) for s in result.sources)
    assert total_suggestions <= 1
