from fastapi import APIRouter, HTTPException, status
from backend.app.schemas import (
    QueryRequest,
    QueryResponse,
    HealthResponse,
    LecturesResponse,
    AddVideoRequest,
    AddVideoResponse
)
from backend.app.services.rag_service import rag_service
from backend.app.services.video_ingestion import video_ingestion_service
from backend.app.services.vector_store import vector_store_service
from backend.app.config import settings

router = APIRouter()

@router.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    """
    Health check endpoint to verify Qdrant connection and service status.
    Useful for Docker healthcheck and load balancers.
    """
    is_qdrant_ok = vector_store_service.check_health()
    return HealthResponse(
        status="healthy" if is_qdrant_ok else "degraded",
        qdrant_connected=is_qdrant_ok,
        collection_name=settings.COLLECTION_NAME,
        models_loaded=True
    )

@router.post("/query", response_model=QueryResponse, tags=["RAG QA"])
def query_dp_knowledge(request: QueryRequest):
    """
    Main RAG QA endpoint:
    1. Embeds user query
    2. Searches top candidates in Qdrant
    3. Re-ranks via Cross-Encoder (bge / ms-marco)
    4. Formats context with exact video timestamps
    5. Generates grounded answer using Groq LLM
    """
    if not request.query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query string cannot be empty."
        )

    try:
        response = rag_service.process_query(request)
        return response
    except Exception as e:
        import traceback
        print(f"[ERROR] RAG pipeline failed: {e}", flush=True)
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error executing RAG pipeline: {str(e)}"
        )

@router.get("/lectures", response_model=LecturesResponse, tags=["Lectures"])
def list_available_lectures():
    """
    Returns list of all indexed Striver DP lectures with titles and video IDs.
    Used by frontend for dropdown filters and information panels.
    """
    try:
        return rag_service.get_lectures()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error loading lectures: {str(e)}"
        )

@router.post("/videos/add", response_model=AddVideoResponse, tags=["Video Ingestion"])
def add_youtube_video(request: AddVideoRequest):
    """
    On-demand incremental video ingestion:
    Protected by Admin Passkey to ensure only the site owner can ingest new lectures.
    """
    # Verify Admin Authorization
    if settings.ADMIN_INGEST_KEY:
        if not request.admin_key or request.admin_key.strip() != settings.ADMIN_INGEST_KEY.strip():
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Unauthorized: Incorrect Admin Passkey. Only the site owner can add new videos."
            )

    if not request.url.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="YouTube URL or video ID cannot be empty."
        )

    try:
        result = video_ingestion_service.ingest_video(
            url_or_id=request.url,
            custom_title=request.title
        )
        return AddVideoResponse(**result)
    except Exception as e:
        import traceback
        print(f"[ERROR] Video ingestion failed: {e}", flush=True)
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to ingest video: {str(e)}"
        )
