"""
Services: FAISSRetriever
Concrete implementation of EmbeddingRetriever using
SentenceTransformer (all-MiniLM-L6-v2) + FAISS IndexFlatIP.
Previously at infrastructure/ml/faiss_retriever.py.
"""
from typing import List, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from app.models.interfaces import EmbeddingRetriever


class FAISSRetriever(EmbeddingRetriever):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self._model = SentenceTransformer(model_name)
        self._index: faiss.Index | None = None

    # ------------------------------------------------------------------ #
    def build_index(self, texts: List[str], batch_size: int = 64) -> None:
        """Encode *texts* and build the FAISS inner-product index."""
        emb = self._model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        emb = np.array(emb, dtype="float32")
        faiss.normalize_L2(emb)
        index = faiss.IndexFlatIP(emb.shape[1])
        index.add(emb)
        self._index = index

    # ── EmbeddingRetriever interface ──────────────────────────────────── #
    def encode_query(self, text: str) -> np.ndarray:
        return self._model.encode(
            [text], normalize_embeddings=True
        ).astype("float32")

    def search(self, query_vector: np.ndarray, k: int) -> Tuple[List[float], List[int]]:
        if self._index is None:
            raise RuntimeError("Index has not been built yet. Call build_index() first.")
        scores_bi, indices = self._index.search(query_vector, k)
        return scores_bi[0].tolist(), indices[0].tolist()
