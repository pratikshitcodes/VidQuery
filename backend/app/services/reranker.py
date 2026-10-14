import time
from typing import List, Tuple, Any
from backend.app.config import settings

class RerankerService:
    def __init__(self):
        self.reranker = None

    def initialize(self):
        """Loads the CrossEncoder model into memory."""
        if self.reranker is None:
            print("⏳ Loading CrossEncoder reranker model...", flush=True)
            from sentence_transformers import CrossEncoder
            self.reranker = CrossEncoder(settings.RERANKER_MODEL_NAME)
            print("✅ Reranker model loaded successfully!", flush=True)

    def rerank(self, query: str, candidates: List[Any], top_k: int = 5) -> Tuple[List[Tuple[Any, float]], float]:
        """
        Reranks retrieved candidate chunks based on full query-passage cross-attention.
        Returns list of (candidate, score) sorted descending, and latency in milliseconds.
        """
        if self.reranker is None:
            self.initialize()

        if not candidates:
            return [], 0.0

        t0 = time.perf_counter()
        pairs = [[query, candidate.payload["text"]] for candidate in candidates]
        scores = self.reranker.predict(pairs)

        ranked = sorted(
            zip(candidates, scores),
            key=lambda x: x[1],
            reverse=True
        )
        rerank_ms = (time.perf_counter() - t0) * 1000.0
        return ranked[:top_k], rerank_ms

reranker_service = RerankerService()
