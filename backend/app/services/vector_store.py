import time
from typing import List, Tuple, Any, Optional
from qdrant_client import QdrantClient
from backend.app.config import settings

class VectorStoreService:
    def __init__(self):
        self.client: QdrantClient = None
        self.embedding_model = None

    def _init_client(self):
        """Initializes the Qdrant client connection if not already connected."""
        if self.client is None:
            self.client = QdrantClient(
                url=settings.QDRANT_URL,
                api_key=settings.QDRANT_API_KEY,
                timeout=15
            )

    def initialize(self):
        """Initializes Qdrant connection and loads FastEmbed ONNX model (~40MB RAM)."""
        self._init_client()
        if self.embedding_model is None:
            print("[LOAD] Initializing FastEmbed (ONNX, ultra-low memory)...", flush=True)
            from fastembed import TextEmbedding
            self.embedding_model = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")
            print("[OK] FastEmbed ONNX model loaded successfully!", flush=True)

    def check_health(self) -> bool:
        """Verifies connection to Qdrant collection without loading heavy ML models."""
        try:
            self._init_client()
            return self.client.collection_exists(settings.COLLECTION_NAME)
        except Exception:
            return False

    def search_candidates(self, query: str, limit: int = 15, video_id: Optional[str] = None) -> Tuple[List[Any], float]:
        """
        Embeds the query using FastEmbed and searches Qdrant for top candidate chunks.
        Optionally filters candidates to a specific video_id.
        Returns candidates and latency in milliseconds.
        """
        if self.client is None or self.embedding_model is None:
            self.initialize()

        t0 = time.perf_counter()
        query_vector = list(self.embedding_model.embed([query]))[0].tolist()
        
        query_filter = None
        if video_id:
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            query_filter = Filter(must=[FieldCondition(key="video_id", match=MatchValue(value=video_id))])

        results = self.client.query_points(
            collection_name=settings.COLLECTION_NAME,
            limit=limit,
            query=query_vector,
            query_filter=query_filter,
            with_payload=True
        )
        qdrant_ms = (time.perf_counter() - t0) * 1000.0
        return results.points, qdrant_ms

vector_store_service = VectorStoreService()
