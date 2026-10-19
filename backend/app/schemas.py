from pydantic import BaseModel, Field
from typing import List, Optional

class QueryRequest(BaseModel):
    query: str = Field(..., description="Technical or conceptual question regarding the DP lectures")
    top_k: Optional[int] = Field(default=5, ge=1, le=10, description="Number of re-ranked context chunks to feed into LLM")
    video_id: Optional[str] = Field(default=None, description="Optional video_id to filter search to a single lecture")

class SourceChunk(BaseModel):
    rank: int
    video_id: str
    lecture_title: str
    start_seconds: int
    end_seconds: int
    start_formatted: str
    end_formatted: str
    timestamp_label: str
    youtube_url: str
    text: str
    rerank_score: float

class LatencyBreakdown(BaseModel):
    qdrant_ms: float
    rerank_ms: float
    llm_ms: float
    total_ms: float

class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: List[SourceChunk]
    latency: LatencyBreakdown

class HealthResponse(BaseModel):
    status: str
    qdrant_connected: bool
    collection_name: str
    models_loaded: bool

class LectureInfo(BaseModel):
    video_id: str
    lecture_title: str
    chunk_count: int
    youtube_url: str

class LecturesResponse(BaseModel):
    total_lectures: int
    lectures: List[LectureInfo]

class AddVideoRequest(BaseModel):
    url: str = Field(..., description="YouTube video URL or 11-character video ID")
    title: Optional[str] = Field(default=None, description="Optional custom lecture title. If omitted, fetched automatically via YouTube.")

class AddVideoResponse(BaseModel):
    status: str
    video_id: str
    lecture_title: str
    chunk_count: int
    total_lectures: int
    message: str
