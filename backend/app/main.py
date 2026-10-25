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

import threading

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan event handler:
    Binds the listening port immediately to pass Render health checks,
    while pre-warming models in the background.
    """
    print("=" * 60, flush=True)
    print("[INIT] Initializing YouTube Video RAG Backend...", flush=True)
    print(f"[QDRANT] Qdrant Target Collection: {settings.COLLECTION_NAME}", flush=True)
    print(f"[MODEL] Embedding Model: {settings.EMBEDDING_MODEL_NAME}", flush=True)
    print(f"[RERANK] Re-Ranker Model: {settings.RERANKER_MODEL_NAME}", flush=True)
    print("=" * 60, flush=True)
    
    # Pre-warm models in background thread so port binds immediately
    threading.Thread(target=rag_service.initialize_models, daemon=True).start()
    print("[OK] Port opened immediately; models pre-warming in background!\n", flush=True)
    yield
    print("[SHUTDOWN] Shutting down YouTube Video RAG Backend...", flush=True)

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

@app.get("/evaluation", tags=["UI"])
def serve_evaluation():
    """Serves the Evaluation & Performance benchmarking dashboard."""
    eval_file = os.path.join(frontend_dir, "evaluation.html")
    if os.path.exists(eval_file):
        return FileResponse(eval_file)
    return FileResponse(os.path.join(frontend_dir, "index.html"))

@app.get("/logo.png", tags=["UI"])
def serve_logo():
    """Serves the VidQuery brand logo."""
    logo_file = os.path.join(frontend_dir, "logo.png")
    if os.path.exists(logo_file):
        return FileResponse(logo_file, media_type="image/png")
    return FileResponse(os.path.join(frontend_dir, "index.html"))

@app.get("/favicon.ico", tags=["UI"])
def serve_favicon():
    """Serves the VidQuery favicon."""
    logo_file = os.path.join(frontend_dir, "logo.png")
    if os.path.exists(logo_file):
        return FileResponse(logo_file, media_type="image/png")
    return FileResponse(os.path.join(frontend_dir, "index.html"))

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

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    print(f"⚡ Starting server on 0.0.0.0:{port}...", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=port)


