"""
Services: QueryMentalHealthSupportService
Orchestrates the full RAG pipeline:
  Stage 1 — Bi-encoder retrieval
  Stage 2 — Text cleaning
  Stage 3 — Cross-encoder re-ranking
  Stage 4 — Deduplication
  Stage 5 — LLM post-processing
  Stage 6 — Source card aggregation

Depends only on model interfaces — no infrastructure imports.
Previously at application/use_cases/query_support.py.
"""
import re
from typing import Optional

from app.models.interfaces import (
    EmbeddingRetriever,
    LLMPostProcessor,
    Reranker,
    SuggestionRepository,
)
from app.models.suggestion import QueryResult, SourceCard, Suggestion


class QueryMentalHealthSupportService:
    def __init__(
        self,
        repository: SuggestionRepository,
        retriever: EmbeddingRetriever,
        reranker: Reranker,
        llm_processor: LLMPostProcessor,
    ) -> None:
        self._repo = repository
        self._retriever = retriever
        self._reranker = reranker
        self._llm = llm_processor

    # ------------------------------------------------------------------ #
    def execute(
        self,
        query: str,
        tag_filter: str = "All",
        n_tips: int = 5,
    ) -> QueryResult:
        df_sug = self._repo.get_suggestion_dataframe()
        concern_meta = self._repo.get_concern_meta()

        # ── Stage 1: Bi-encoder retrieval ──────────────────────────────
        fetch_k = 50 if tag_filter != "All" else 30
        q_vec = self._retriever.encode_query(query)
        scores, indices = self._retriever.search(q_vec, fetch_k)

        if not indices:
            return QueryResult(answer="No relevant results found.")

        retrieved = df_sug.iloc[indices].copy()
        retrieved["_score"] = scores

        if tag_filter != "All":
            retrieved = retrieved[retrieved["tag"] == tag_filter]
        if retrieved.empty:
            return QueryResult(answer="No results found under that filter. Try 'All'.")

        # ── Stage 2: Text cleaning ─────────────────────────────────────
        candidates: list[Suggestion] = []
        for _, row in retrieved.iterrows():
            text = re.sub(r"\{\{.*?\}\}", "", row["text"])
            text = re.sub(r"\s+", " ", text).strip()
            if len(text) < 30:
                continue
            candidates.append(
                Suggestion(
                    text=text,
                    tag=row["tag"],
                    labels=row["multi_labels"],
                    concern=row["Concern"],
                )
            )

        if not candidates:
            return QueryResult(answer="I couldn't find relevant suggestions for your query.")

        # ── Stage 3: Cross-encoder re-ranking ─────────────────────────
        texts = [c.text for c in candidates]
        ce_scores = self._reranker.rerank(query, texts)
        for c, score in zip(candidates, ce_scores):
            c.ce_score = float(score)
        candidates.sort(key=lambda c: c.ce_score, reverse=True)

        # ── Stage 4: Deduplication ─────────────────────────────────────
        seen: set[str] = set()
        top_tips: list[Suggestion] = []
        for c in candidates:
            key = c.text[:60].lower()
            if key in seen:
                continue
            seen.add(key)
            top_tips.append(c)
            if len(top_tips) >= n_tips:
                break

        if not top_tips:
            return QueryResult(answer="No suggestions found.")

        # ── Stage 5: LLM post-processing ──────────────────────────────
        answer = self._llm.generate(query, [t.text for t in top_tips])

        # ── Stage 6: Source card aggregation ──────────────────────────
        by_concern: dict[str, list[str]] = {}
        for tip in top_tips:
            by_concern.setdefault(tip.concern, []).append(tip.text)

        sources: list[SourceCard] = []
        for concern_text, tips in by_concern.items():
            row = concern_meta[concern_meta["Concern"] == concern_text]
            if row.empty:
                continue
            sources.append(
                SourceCard(
                    concern=concern_text,
                    title=row["title"].values[0],
                    tag=row["tag"].values[0],
                    all_labels=row["all_labels"].values[0],
                    suggestions=tips,
                )
            )

        return QueryResult(answer=answer, sources=sources)
