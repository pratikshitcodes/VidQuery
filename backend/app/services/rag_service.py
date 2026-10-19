import os
import json
import re
from typing import List, Dict, Any
from backend.app.config import settings
from backend.app.schemas import (
    QueryRequest,
    QueryResponse,
    SourceChunk,
    LatencyBreakdown,
    LectureInfo,
    LecturesResponse
)
from backend.app.services.vector_store import vector_store_service
from backend.app.services.reranker import reranker_service
from backend.app.services.llm import llm_service

def format_seconds_to_mmss(seconds: float) -> str:
    """Converts seconds float/int to MM:SS string."""
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"

def extract_lecture_title(chunk_text: str, default_id: str) -> str:
    """Extracts title from '[Video: DP X: Title]' header if present."""
    match = re.search(r"\[Video:\s*([^\]]+)\]", chunk_text)
    if match:
        return match.group(1).strip()
    return f"Lecture {default_id}"

class RAGService:
    def __init__(self):
        self._lectures_cache: List[LectureInfo] = None

    def initialize_models(self):
        """Pre-loads vector store, embedding model, and re-ranker on server startup."""
        vector_store_service.initialize()
        reranker_service.initialize()
        llm_service.initialize()

    def process_query(self, request: QueryRequest) -> QueryResponse:
        """
        Executes two-stage retrieval (Qdrant + CrossEncoder) and generates
        a grounded answer with clickable timestamp metadata.
        """
        query = request.query.strip()
        top_k = request.top_k or settings.DEFAULT_TOP_K

        # Step 1: Qdrant dense retrieval
        candidates, qdrant_ms = vector_store_service.search_candidates(
            query=query,
            limit=settings.DENSE_SEARCH_LIMIT,
            video_id=request.video_id
        )

        # Step 2: Cross-Encoder re-ranking
        ranked_pairs, rerank_ms = reranker_service.rerank(
            query=query,
            candidates=candidates,
            top_k=top_k
        )

        # Step 3: Build SourceChunk list & LLM context
        sources: List[SourceChunk] = []
        context_blocks: List[str] = []

        for rank, (point, score) in enumerate(ranked_pairs, 1):
            payload = point.payload
            start_s = int(payload.get("start", 0))
            end_s = int(payload.get("end", 0))
            vid = payload.get("video_id", "unknown")
            text = payload.get("text", "")
            title = extract_lecture_title(text, vid)
            start_fmt = format_seconds_to_mmss(start_s)
            end_fmt = format_seconds_to_mmss(end_s)
            timestamp_label = f"[{start_fmt} - {end_fmt}]"
            yt_url = f"https://www.youtube.com/watch?v={vid}&t={start_s}s"

            sources.append(
                SourceChunk(
                    rank=rank,
                    video_id=vid,
                    lecture_title=title,
                    start_seconds=start_s,
                    end_seconds=end_s,
                    start_formatted=start_fmt,
                    end_formatted=end_fmt,
                    timestamp_label=timestamp_label,
                    youtube_url=yt_url,
                    text=text,
                    rerank_score=round(float(score), 4)
                )
            )

            context_blocks.append(
                f"{timestamp_label} (Video: {title})\n{text}"
            )

        context_str = "\n\n".join(context_blocks)

        # Step 4: Groq LLM generation
        answer, llm_ms = llm_service.generate_answer(context=context_str, query=query)

        total_ms = qdrant_ms + rerank_ms + llm_ms

        return QueryResponse(
            query=query,
            answer=answer,
            sources=sources,
            latency=LatencyBreakdown(
                qdrant_ms=round(qdrant_ms, 1),
                rerank_ms=round(rerank_ms, 1),
                llm_ms=round(llm_ms, 1),
                total_ms=round(total_ms, 1)
            )
        )

    def get_lectures(self) -> LecturesResponse:
        """Returns metadata for all 31 Striver DP lectures from data/chunks/."""
        if self._lectures_cache is not None:
            return LecturesResponse(total_lectures=len(self._lectures_cache), lectures=self._lectures_cache)

        lectures_list: List[LectureInfo] = []
        chunks_dir = settings.CHUNKS_DIR

        if os.path.exists(chunks_dir):
            for filename in sorted(os.listdir(chunks_dir)):
                if filename.endswith(".json"):
                    vid = os.path.splitext(filename)[0]
                    file_path = os.path.join(chunks_dir, filename)
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            chunks = json.load(f)
                        count = len(chunks)
                        title = extract_lecture_title(chunks[0]["text"], vid) if chunks else f"Lecture {vid}"
                        yt_url = f"https://www.youtube.com/watch?v={vid}"
                        lectures_list.append(
                            LectureInfo(
                                video_id=vid,
                                lecture_title=title,
                                chunk_count=count,
                                youtube_url=yt_url
                            )
                        )
                    except Exception:
                        continue

        self._lectures_cache = lectures_list
        return LecturesResponse(total_lectures=len(lectures_list), lectures=lectures_list)

rag_service = RAGService()
