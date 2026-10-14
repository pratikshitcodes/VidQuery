import sys
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from backend.app.config import settings
from backend.app.api.routes import router as api_router, health_check
from backend.app.services.rag_service import rag_service

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan event handler:
    Pre-warms embedding models, CrossEncoder, and database connections
    on application startup so first user query has zero cold-start delay.
    """
    print("=" * 60)
    print("🚀 Initializing YouTube Video RAG Backend...")
    print(f"📦 Qdrant Target Collection: {settings.COLLECTION_NAME}")
    print(f"🧠 Embedding Model: {settings.EMBEDDING_MODEL_NAME}")
    print(f"🎯 Re-Ranker Model: {settings.RERANKER_MODEL_NAME}")
    print("=" * 60)
    
    rag_service.initialize_models()
    print("✅ Models and Vector Store initialized successfully!\n")
    yield
    print("🛑 Shutting down YouTube Video RAG Backend...")

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Production-grade Video RAG backend indexing 31 Striver Dynamic Programming YouTube lectures with exact second-level deep-linking and Cross-Encoder re-ranking.",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API router
app.include_router(api_router, prefix=settings.API_V1_STR)

# Also expose health check at top-level /health
@app.get("/health", tags=["Health"])
def root_health():
    return health_check()

# Serve Frontend SPA
frontend_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "frontend"
)

if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

@app.get("/", tags=["UI"])
def serve_ui():
    """Serves the interactive web UI with synchronized YouTube video player."""
    index_file = os.path.join(frontend_dir, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {
        "project": settings.PROJECT_NAME,
        "docs_url": "/docs",
        "health": "/health"
    }

@app.get("/api", tags=["Root"])
def root_info():
    return {
        "project": settings.PROJECT_NAME,
        "version": "1.0.0",
        "docs_url": "/docs",
        "health_check": "/health",
        "endpoints": {
            "query": f"{settings.API_V1_STR}/query",
            "lectures": f"{settings.API_V1_STR}/lectures",
            "health": f"{settings.API_V1_STR}/health"
        }
    }

