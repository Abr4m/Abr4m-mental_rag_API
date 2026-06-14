"""
Database: AppContainer
Builds and holds all singleton infrastructure instances.
Wired once during the FastAPI lifespan event, then injected via dependency.

Renamed from core/container.py — the container manages data/model resources
that are conceptually analogous to database connections in standard projects.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI

from app.core.config import get_settings
from app.models.interfaces import (
    EmbeddingRetriever,
    LLMPostProcessor,
    Reranker,
    SuggestionRepository,
)
from app.repositories.csv_repository import CSVSuggestionRepository
from app.services.cross_encoder_reranker import CrossEncoderReranker
from app.services.faiss_retriever import FAISSRetriever
from app.services.qwen_processor import FallbackLLMPostProcessor, QwenLLMPostProcessor

logger = logging.getLogger(__name__)


class AppContainer:
    """Dependency-injection container: one instance per process."""

    repository: SuggestionRepository
    retriever: EmbeddingRetriever
    reranker: Reranker
    llm_processor: LLMPostProcessor

    def __init__(self) -> None:
        settings = get_settings()
        logger.info("Initialising AppContainer …")

        # ── Repository ────────────────────────────────────────────────
        self.repository = CSVSuggestionRepository(settings.csv_path)
        df_sug = self.repository.get_suggestion_dataframe()
        logger.info("Dataset loaded: %d suggestion rows", len(df_sug))

        # ── Retriever + FAISS index ───────────────────────────────────
        self.retriever = FAISSRetriever(settings.bi_encoder_model)
        logger.info("Building FAISS index …")
        self.retriever.build_index(df_sug["text"].tolist())  # type: ignore[union-attr]
        logger.info("FAISS index built.")

        # ── Reranker ──────────────────────────────────────────────────
        self.reranker = CrossEncoderReranker(settings.cross_encoder_model)
        logger.info("CrossEncoder loaded.")

        # ── LLM post-processor ────────────────────────────────────────
        if settings.use_fallback_llm:
            logger.warning("use_fallback_llm=True — skipping Qwen model load.")
            self.llm_processor = FallbackLLMPostProcessor()
        else:
            logger.info("Loading Qwen LLM …")
            self.llm_processor = QwenLLMPostProcessor(
                settings.llm_model, settings.llm_max_new_tokens
            )
            logger.info("Qwen LLM ready.")

        logger.info("AppContainer ready.")


# Module-level singleton — populated during lifespan startup
_container: AppContainer | None = None


def get_container() -> AppContainer:
    if _container is None:
        raise RuntimeError("AppContainer has not been initialised yet.")
    return _container


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global _container
    _container = AppContainer()
    yield
    _container = None
