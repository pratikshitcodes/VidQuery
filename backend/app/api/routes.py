from fastapi import APIRouter, HTTPException, status
from backend.app.schemas import (
    QueryRequest,
    QueryResponse,
    HealthResponse,
    LecturesResponse
)
from backend.app.services.rag_service import rag_service
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
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error executing RAG pipeline: {str(e)}"
        )

@router.get("/lectures", response_model=LecturesResponse, tags=["Lectures"])
def list_available_lectures():
    """
    Returns list of all 31 indexed Striver DP lectures with titles and video IDs.
    Used by frontend for dropdown filters and information panels.
    """
    try:
        return rag_service.get_lectures()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error loading lectures: {str(e)}"
        )
