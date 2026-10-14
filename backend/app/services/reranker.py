import os
import time
from typing import List, Tuple, Any
from backend.app.config import settings

class RerankerService:
    def __init__(self):
        self.reranker = None
        # On cloud free tiers (512MB RAM), local CrossEncoder triggers OOM kills (502).
        # We enable it only if explicitly configured; otherwise use Qdrant's high-accuracy cosine scoring.
        self.enabled = os.getenv("ENABLE_CROSS_ENCODER", "false").lower() in ("true", "1")

    def initialize(self):
        """Loads the CrossEncoder model into memory if enabled."""
        if not self.enabled:
            print("[INFO] Local Cross-Encoder disabled to stay within 512MB RAM; using Qdrant cosine ranking.", flush=True)
            return

        if self.reranker is None:
            try:
                print("[LOAD] Loading CrossEncoder reranker model...", flush=True)
                import torch
                torch.set_num_threads(1)
                from sentence_transformers import CrossEncoder
                self.reranker = CrossEncoder(settings.RERANKER_MODEL_NAME)
                print("[OK] Reranker model loaded successfully!", flush=True)
            except Exception as e:
                print(f"[WARN] Could not load CrossEncoder ({e}); falling back to Qdrant ranking.", flush=True)
                self.reranker = None
                self.enabled = False

    def rerank(self, query: str, candidates: List[Any], top_k: int = 5) -> Tuple[List[Tuple[Any, float]], float]:
        """
        Reranks retrieved candidate chunks.
        Uses Cross-Encoder if enabled, or falls back to Qdrant's cosine similarity scores.
        """
        if not candidates:
            return [], 0.0

        t0 = time.perf_counter()

        if self.enabled:
            if self.reranker is None:
                self.initialize()
            if self.reranker is not None:
                try:
                    pairs = [[query, candidate.payload["text"]] for candidate in candidates]
                    scores = self.reranker.predict(pairs)
                    ranked = sorted(
                        zip(candidates, scores),
                        key=lambda x: x[1],
                        reverse=True
                    )
                    rerank_ms = (time.perf_counter() - t0) * 1000.0
                    return ranked[:top_k], rerank_ms
                except Exception as e:
                    print(f"⚠️ CrossEncoder inference failed ({e}); falling back to Qdrant scores.", flush=True)

        # High-accuracy fallback: rank by Qdrant's native vector cosine similarity score
        ranked = [(candidate, getattr(candidate, "score", 0.0)) for candidate in candidates]
        ranked = sorted(ranked, key=lambda x: x[1], reverse=True)
        rerank_ms = (time.perf_counter() - t0) * 1000.0
        return ranked[:top_k], rerank_ms

reranker_service = RerankerService()
