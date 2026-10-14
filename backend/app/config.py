import os
from dotenv import load_dotenv

# Load from root .env or environment
load_dotenv()

class Settings:
    PROJECT_NAME: str = "YouTube Video RAG - Striver DP Reviser"
    API_V1_STR: str = "/api/v1"
    
    # Qdrant settings
    QDRANT_URL: str = os.getenv("QDRANT_URL", "")
    QDRANT_API_KEY: str = os.getenv("QDRANT_API_KEY", "")
    COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "knowledge")
    EMBEDDING_SIZE: int = 384
    
    # Models
    EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
    RERANKER_MODEL_NAME: str = os.getenv("RERANKER_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    
    # Groq LLM
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    LLM_MODEL_NAME: str = os.getenv("LLM_MODEL_NAME", "openai/gpt-oss-120b")
    
    # Retrieval Hyperparameters
    DENSE_SEARCH_LIMIT: int = 15
    DEFAULT_TOP_K: int = 5
    
    # Data directory
    CHUNKS_DIR: str = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "data",
        "chunks"
    )

settings = Settings()
