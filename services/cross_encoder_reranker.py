"""
Services: CrossEncoderReranker
Concrete implementation of Reranker using
sentence-transformers CrossEncoder (ms-marco-MiniLM-L-6-v2).
Previously at infrastructure/ml/cross_encoder_reranker.py.
"""
from typing import List

from sentence_transformers import CrossEncoder

from app.models.interfaces import Reranker


class CrossEncoderReranker(Reranker):
    def __init__(
        self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ) -> None:
        self._model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: List[str]) -> List[float]:
        pairs = [(query, c) for c in candidates]
        scores = self._model.predict(pairs, show_progress_bar=False)
        return [float(s) for s in scores]
