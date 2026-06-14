"""
Models: Domain Interfaces (Ports)
Abstract base classes defining contracts between the application layer
and all infrastructure adapters. No framework or ML dependencies here.
"""
from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np
import pandas as pd


# ── Repository ────────────────────────────────────────────────────────────────

class SuggestionRepository(ABC):
    @abstractmethod
    def get_suggestion_dataframe(self) -> pd.DataFrame:
        """Return DataFrame with columns: text, tag, multi_labels, Concern."""

    @abstractmethod
    def get_concern_meta(self) -> pd.DataFrame:
        """Return DataFrame with columns: Concern, tag, title, all_labels."""

    @abstractmethod
    def get_available_tags(self) -> List[str]:
        """Return sorted list of unique tag values (plus 'All')."""


# ── Retriever ─────────────────────────────────────────────────────────────────

class EmbeddingRetriever(ABC):
    @abstractmethod
    def encode_query(self, text: str) -> np.ndarray:
        """Return a normalised float32 embedding vector for *text*."""

    @abstractmethod
    def search(self, query_vector: np.ndarray, k: int) -> Tuple[List[float], List[int]]:
        """Return (scores, indices) for the top-k nearest suggestion vectors."""


# ── Reranker ──────────────────────────────────────────────────────────────────

class Reranker(ABC):
    @abstractmethod
    def rerank(self, query: str, candidates: List[str]) -> List[float]:
        """Return a float score per candidate (higher = more relevant)."""


# ── LLM Post-Processor ────────────────────────────────────────────────────────

class LLMPostProcessor(ABC):
    @abstractmethod
    def generate(self, query: str, suggestions: List[str]) -> str:
        """Summarise and structure *suggestions* relative to *query*."""
