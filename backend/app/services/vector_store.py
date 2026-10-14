import time
from typing import List, Tuple, Any
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from backend.app.config import settings

class VectorStoreService:
    def __init__(self):
        self.client: QdrantClient = None
        self.embedding_model: SentenceTransformer = None

    def initialize(self):
        """Initializes Qdrant connection and loads the embedding model into memory."""
        if self.client is None:
            self.client = QdrantClient(
                url=settings.QDRANT_URL,
                api_key=settings.QDRANT_API_KEY,
                timeout=60
            )
        if self.embedding_model is None:
            self.embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)

    def check_health(self) -> bool:
        """Verifies connection to Qdrant collection."""
        try:
            if self.client is None:
                self.initialize()
            return self.client.collection_exists(settings.COLLECTION_NAME)
        except Exception:
            return False

    def search_candidates(self, query: str, limit: int = 15) -> Tuple[List[Any], float]:
        """
        Embeds the query and searches Qdrant for top candidate chunks.
        Returns candidates and latency in milliseconds.
        """
        if self.client is None or self.embedding_model is None:
            self.initialize()

        t0 = time.perf_counter()
        query_vector = self.embedding_model.encode(query).tolist()
        
        results = self.client.query_points(
            collection_name=settings.COLLECTION_NAME,
            limit=limit,
            query=query_vector,
            with_payload=True
        )
        qdrant_ms = (time.perf_counter() - t0) * 1000.0
        return results.points, qdrant_ms

vector_store_service = VectorStoreService()
