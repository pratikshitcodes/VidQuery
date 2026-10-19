import os
import re
import json
import uuid
import time
import urllib.request
from typing import Dict, Any, List, Optional
from qdrant_client.models import PointStruct

from backend.app.config import settings
from backend.app.services.vector_store import vector_store_service
from backend.app.services.rag_service import rag_service, extract_lecture_title

CHUNK_WINDOW_SECONDS = 55.0
CHUNK_OVERLAP_SECONDS = 15.0

class VideoIngestionService:
    def __init__(self):
        self.chunks_dir = settings.CHUNKS_DIR
        self.raw_dir = os.path.join(os.path.dirname(self.chunks_dir), "raw_transcripts")
        os.makedirs(self.chunks_dir, exist_ok=True)
        os.makedirs(self.raw_dir, exist_ok=True)

    def extract_video_id(self, url_or_id: str) -> str:
        """Extracts standard 11-char YouTube video ID from various URL patterns or raw ID."""
        text = url_or_id.strip()
        patterns = [
            r"(?:v=|\/embed\/|\/watch\?v=|youtu\.be\/|\/v\/)([a-zA-Z0-9_-]{11})",
            r"^([a-zA-Z0-9_-]{11})$"
        ]
        for pat in patterns:
            match = re.search(pat, text)
            if match:
                return match.group(1)
        raise ValueError(f"Could not extract a valid 11-character YouTube video ID from: '{url_or_id}'")

    def fetch_video_title(self, video_id: str) -> str:
        """Fetches video title using YouTube's public oEmbed API in <50ms without API keys."""
        try:
            oembed_url = f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={video_id}&format=json"
            req = urllib.request.Request(oembed_url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=8) as r:
                data = json.loads(r.read().decode("utf-8"))
                title = data.get("title", "").strip()
                if title:
                    return title
        except Exception as e:
            print(f"[WARN] Failed to fetch oEmbed title for {video_id}: {e}", flush=True)
        return f"Lecture {video_id}"

    def fetch_transcript_segments(self, video_id: str) -> List[Dict[str, Any]]:
        """
        Fetches timed transcript segments for a video.
        Uses YouTubeTranscriptApi for ultra-fast, zero-overhead fetching (<2s).
        Falls back to yt-dlp + whisper only if human/auto captions are disabled.
        """
        raw_path = os.path.join(self.raw_dir, f"{video_id}.json")
        if os.path.exists(raw_path):
            with open(raw_path, "r", encoding="utf-8") as f:
                return json.load(f)

        # 1. Primary: YouTube official / auto-generated transcript API
        try:
            from youtube_transcript_api import YouTubeTranscriptApi
            api = YouTubeTranscriptApi()
            transcript = api.fetch(video_id, languages=("en", "hi", "en-IN", "en-US", "en-GB"))
            segments = []
            for item in transcript:
                start_s = round(float(item.start), 2)
                end_s = round(float(item.start + item.duration), 2)
                text = item.text.strip()
                if text:
                    segments.append({
                        "start": start_s,
                        "end": end_s,
                        "text": text
                    })
            if segments:
                with open(raw_path, "w", encoding="utf-8") as f:
                    json.dump(segments, f, ensure_ascii=False, indent=2)
                print(f"[INGEST] Fetched {len(segments)} segments via YouTube captions API for {video_id}", flush=True)
                return segments
        except Exception as e:
            print(f"[WARN] YouTube captions API failed for {video_id}: {e}", flush=True)

        # 2. Fallback: Download audio and run whisper if available
        try:
            import yt_dlp
            audio_dir = os.path.join(os.path.dirname(self.chunks_dir), "audio")
            os.makedirs(audio_dir, exist_ok=True)
            out_file = os.path.join(audio_dir, f"{video_id}.webm")
            
            if not os.path.exists(out_file):
                ydl_opts = {
                    "format": "bestaudio/best",
                    "outtmpl": os.path.join(audio_dir, "%(id)s.%(ext)s"),
                    "quiet": True,
                    "no_warnings": True
                }
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([f"https://www.youtube.com/watch?v={video_id}"])

            try:
                import whisper
                model = whisper.load_model("tiny")
                res = model.transcribe(out_file)
                segments = [
                    {
                        "start": round(float(s["start"]), 2),
                        "end": round(float(s["end"]), 2),
                        "text": s["text"].strip()
                    }
                    for s in res.get("segments", []) if s["text"].strip()
                ]
                if segments:
                    with open(raw_path, "w", encoding="utf-8") as f:
                        json.dump(segments, f, ensure_ascii=False, indent=2)
                    return segments
            except ImportError:
                pass
        except Exception as e:
            print(f"[ERROR] Audio download/whisper fallback failed: {e}", flush=True)

        raise RuntimeError(f"Could not retrieve captions or transcripts for YouTube video '{video_id}'. Ensure captions/subtitles are enabled on the video.")

    def chunk_segments(self, segments: List[Dict[str, Any]], video_id: str, lecture_title: str) -> List[Dict[str, Any]]:
        """Splits transcript segments into 55s overlapping contextual chunks with lecture headers."""
        if not segments:
            return []

        chunks = []
        n = len(segments)
        idx = 0

        while idx < n:
            chunk_start = segments[idx]["start"]
            current_segs = []
            j = idx

            while j < n:
                current_segs.append(segments[j])
                if segments[j]["end"] - chunk_start >= CHUNK_WINDOW_SECONDS:
                    j += 1
                    break
                j += 1

            chunk_end = current_segs[-1]["end"]
            raw_body = " ".join(s["text"].strip() for s in current_segs if s["text"].strip())
            contextual_text = f"[Video: {lecture_title}] {raw_body}"

            chunks.append({
                "start": chunk_start,
                "end": chunk_end,
                "text": contextual_text,
                "video_id": video_id
            })

            if j >= n:
                break

            target_next_start = chunk_end - CHUNK_OVERLAP_SECONDS
            next_idx = idx + 1
            for k in range(idx + 1, j):
                if segments[k]["start"] >= target_next_start:
                    next_idx = k
                    break
            else:
                next_idx = j

            idx = max(next_idx, idx + 1)

        return chunks

    def ingest_video(self, url_or_id: str, custom_title: Optional[str] = None) -> Dict[str, Any]:
        """
        Incrementally processes and adds a single new YouTube video to the RAG knowledge base:
        1. Checks if video is already indexed -> skips immediately if present.
        2. Fetches metadata and timed transcript.
        3. Creates contextual chunks.
        4. Embeds via FastEmbed ONNX.
        5. Upserts points into Qdrant collection with deterministic UUIDs.
        6. Saves local chunk cache & updates lecture list.
        """
        video_id = self.extract_video_id(url_or_id)
        chunk_file = os.path.join(self.chunks_dir, f"{video_id}.json")

        # Check if already indexed locally
        if os.path.exists(chunk_file):
            with open(chunk_file, "r", encoding="utf-8") as f:
                existing_chunks = json.load(f)
            title = custom_title or (
                extract_lecture_title(existing_chunks[0]["text"], video_id)
                if existing_chunks else f"Lecture {video_id}"
            )
            rag_service._lectures_cache = None
            total_lectures = rag_service.get_lectures().total_lectures
            return {
                "status": "already_indexed",
                "video_id": video_id,
                "lecture_title": title,
                "chunk_count": len(existing_chunks),
                "total_lectures": total_lectures,
                "message": f"'{title}' is already indexed ({len(existing_chunks)} chunks). Ready for search!"
            }

        print(f"[INGEST] Processing new video: {video_id}...", flush=True)

        # 1. Title
        title = custom_title.strip() if custom_title and custom_title.strip() else self.fetch_video_title(video_id)
        
        # 2. Transcript
        segments = self.fetch_transcript_segments(video_id)

        # 3. Chunking
        chunks = self.chunk_segments(segments, video_id, title)
        if not chunks:
            raise ValueError(f"No valid transcript content found in video '{video_id}'.")

        # 4. Save chunks locally
        with open(chunk_file, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)

        # 5. Embed with FastEmbed
        vector_store_service.initialize()
        texts = [c["text"] for c in chunks]
        embeddings = list(vector_store_service.embedding_model.embed(texts))

        # 6. Upsert to Qdrant Cloud with deterministic UUIDs
        points = []
        for i, chunk in enumerate(chunks):
            pt_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{video_id}_{i}"))
            points.append(
                PointStruct(
                    id=pt_id,
                    vector=embeddings[i].tolist(),
                    payload={
                        "start": chunk["start"],
                        "end": chunk["end"],
                        "text": chunk["text"],
                        "video_id": video_id
                    }
                )
            )

        client = vector_store_service.client
        BATCH_SIZE = 100
        for start_idx in range(0, len(points), BATCH_SIZE):
            batch = points[start_idx : start_idx + BATCH_SIZE]
            client.upsert(collection_name=settings.COLLECTION_NAME, points=batch)

        print(f"[INGEST-OK] Indexed {len(points)} chunks for '{title}' into Qdrant Cloud!", flush=True)

        # Invalidate lecture cache so new lecture appears immediately
        rag_service._lectures_cache = None
        total_lectures = rag_service.get_lectures().total_lectures

        return {
            "status": "success",
            "video_id": video_id,
            "lecture_title": title,
            "chunk_count": len(chunks),
            "total_lectures": total_lectures,
            "message": f"Successfully indexed '{title}' ({len(chunks)} chunks). Available for search immediately!"
        }

video_ingestion_service = VideoIngestionService()
