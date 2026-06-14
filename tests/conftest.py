"""
Tests: Shared Stubs & Fixtures
All infrastructure is mocked here — no models are loaded during tests.
"""
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.models.interfaces import (
    EmbeddingRetriever,
    LLMPostProcessor,
    Reranker,
    SuggestionRepository,
)


# ── Stubs ─────────────────────────────────────────────────────────────────────

class StubRepository(SuggestionRepository):
    def get_suggestion_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "text": [
                    "Try deep breathing exercises daily.",
                    "Talk to a trusted friend about your feelings.",
                    "Consider journaling to process your emotions.",
                ],
                "tag": ["anxiety", "depression", "anxiety"],
                "multi_labels": ["coping_strategy", "personal_experience", "coping_strategy"],
                "Concern": ["Concern A", "Concern B", "Concern A"],
            }
        )

    def get_concern_meta(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "Concern": ["Concern A", "Concern B"],
                "tag": ["anxiety", "depression"],
                "title": ["Title A", "Title B"],
                "all_labels": ["coping_strategy", "personal_experience"],
            }
        )

    def get_available_tags(self):
        return ["All", "anxiety", "depression"]


class StubRetriever(EmbeddingRetriever):
    def encode_query(self, text: str) -> np.ndarray:
        return np.zeros((1, 384), dtype="float32")

    def search(self, query_vector, k: int):
        return [0.9, 0.8, 0.7], [0, 1, 2]


class StubReranker(Reranker):
    def rerank(self, query: str, candidates):
        return [1.0, 0.9, 0.8][: len(candidates)]


class StubLLM(LLMPostProcessor):
    def generate(self, query: str, suggestions) -> str:
        return "Stub answer."


# ── App fixture ───────────────────────────────────────────────────────────────

@pytest.fixture
def client(monkeypatch):
    import app.database.container as container_module

    class StubContainer:
        repository = StubRepository()
        retriever = StubRetriever()
        reranker = StubReranker()
        llm_processor = StubLLM()

    monkeypatch.setattr(container_module, "_container", StubContainer())

    from app.main import create_app
    test_app = create_app()
    with TestClient(test_app, raise_server_exceptions=True) as c:
        yield c
