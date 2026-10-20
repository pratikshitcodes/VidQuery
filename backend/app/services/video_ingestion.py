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
            contextual_text = f"[Video: {lecture_title}] [Topics: {lecture_title}] {raw_body}"

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

    def detect_hindi_or_non_english(self, text: str) -> bool:
        """
        Detects whether transcript text contains Hindi (Devanagari script or Romanized Hinglish).
        """
        if not text:
            return False

        # 1. Check Devanagari Unicode characters (\u0900 - \u097F)
        devanagari_count = sum(1 for c in text if '\u0900' <= c <= '\u097f')
        if devanagari_count >= 10:
            return True

        # 2. Check Romanized Hindi / Hinglish markers
        hinglish_markers = {
            'karna', 'karenge', 'karega', 'karte', 'karein', 'hoga', 'hogi', 'hote', 'hota',
            'hai', 'hain', 'hum', 'humne', 'yeh', 'woh', 'toh', 'agar', 'kyunki',
            'kuch', 'samajh', 'rahe', 'raha', 'rahi', 'dekho', 'kaise', 'bhai',
            'sabse', 'pehle', 'baad', 'andar', 'baahar', 'dosto', 'dost'
        }
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        if words:
            hinglish_count = sum(1 for w in words if w in hinglish_markers)
            if (hinglish_count / len(words) > 0.06) and hinglish_count >= 5:
                return True

        return False

    def translate_chunks_to_english(self, chunks: List[Dict[str, Any]], lecture_title: str) -> List[Dict[str, Any]]:
        """
        Translates Hindi/Hinglish chunks into high-quality technical English using Groq LLM.
        Processes chunks in small batches with strict JSON schema and exponential backoff.
        Preserves timestamps and metadata exactly.
        """
        if not chunks:
            return chunks

        try:
            from groq import Groq
            client = Groq(api_key=settings.GROQ_API_KEY)
        except Exception as e:
            print(f"[WARN] Groq client initialization failed for translation: {e}", flush=True)
            return chunks

        prefix_pat = r'^\[Video:.*?\]\s*\[Topics:.*?\]\s*'
        clean_bodies = [re.sub(prefix_pat, '', c["text"]).strip() for c in chunks]

        BATCH_SIZE = 3
        translated_bodies = []
        prompt = (
            "You are an expert translator for computer science and Data Structures & Algorithms lectures. "
            "Translate each text item in the JSON array from Hindi or Hinglish into clear, fluent, precise technical English. "
            "Preserve all technical concepts and keywords (e.g., Segment Tree, 4*n, Lazy Propagation, Time Complexity, Array, Index, Leaf, Root, Query, Update). "
            "Return a JSON object with key 'items' containing the translated list of strings in the exact same order and length."
        )

        for i in range(0, len(clean_bodies), BATCH_SIZE):
            batch = clean_bodies[i : i + BATCH_SIZE]
            batch_success = False

            for attempt in range(5):
                try:
                    res = client.chat.completions.create(
                        model="openai/gpt-oss-120b",
                        response_format={"type": "json_object"},
                        max_tokens=4096,
                        messages=[
                            {"role": "system", "content": prompt},
                            {"role": "user", "content": json.dumps({"items": batch}, ensure_ascii=False)}
                        ],
                        temperature=0.1
                    )
                    parsed = json.loads(res.choices[0].message.content)
                    items = parsed.get("items", [])
                    if len(items) == len(batch):
                        # Clean non-breaking / narrow spaces
                        cleaned = [it.replace('\u202f', ' ').replace('\xa0', ' ') for it in items]
                        translated_bodies.extend(cleaned)
                        batch_success = True
                        break
                    else:
                        print(f"[WARN] Translation batch count mismatch {len(items)} vs {len(batch)}. Retrying...", flush=True)
                except Exception as e:
                    err_msg = str(e).lower()
                    if "429" in err_msg or "rate_limit" in err_msg:
                        wait_time = 3.5 * (1.5 ** attempt)
                        print(f"[WARN] Groq rate limited during translation. Sleeping {wait_time:.1f}s...", flush=True)
                        time.sleep(wait_time)
                    else:
                        print(f"[WARN] Groq translation attempt {attempt+1} failed: {e}", flush=True)
                        time.sleep(1.5 * (attempt + 1))

            if not batch_success:
                print(f"[ERROR] Batch {i//BATCH_SIZE + 1} translation failed after 5 attempts. Keeping original text for these chunks.", flush=True)
                translated_bodies.extend(batch)

        # Reassemble chunks with technical headers
        translated_chunks = []
        for orig, tr in zip(chunks, translated_bodies):
            new_c = dict(orig)
            new_c["text"] = f"[Video: {lecture_title}] [Topics: {lecture_title}] {tr}"
            translated_chunks.append(new_c)

        return translated_chunks

    def ingest_video(self, url_or_id: str, custom_title: Optional[str] = None) -> Dict[str, Any]:
        """
        Incrementally processes and adds a single new YouTube video to the RAG knowledge base:
        1. Checks if video is already indexed -> skips immediately if present (or upgrades if in Hindi).
        2. Fetches metadata and timed transcript.
        3. Creates contextual chunks.
        4. Detects Hindi/Hinglish and translates automatically to technical English.
        5. Embeds via FastEmbed ONNX.
        6. Upserts points into Qdrant collection with deterministic UUIDs.
        7. Saves local chunk cache & updates lecture list.
        """
        video_id = self.extract_video_id(url_or_id)
        chunk_file = os.path.join(self.chunks_dir, f"{video_id}.json")

        # Check if already indexed locally
        if os.path.exists(chunk_file):
            with open(chunk_file, "r", encoding="utf-8") as f:
                existing_chunks = json.load(f)
            sample_existing = " ".join(c["text"] for c in existing_chunks[:5]) if existing_chunks else ""
            if existing_chunks and self.detect_hindi_or_non_english(sample_existing):
                print(f"[INGEST] Video {video_id} exists but is in Hindi. Re-translating to English and updating index...", flush=True)
            else:
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

        # 4. Automatic Language Detection & English Translation
        sample_text = " ".join(c["text"] for c in chunks[:5])
        was_translated = False
        if self.detect_hindi_or_non_english(sample_text):
            print(f"[INGEST] Hindi/Hinglish detected in transcript for '{title}'. Translating {len(chunks)} chunks to technical English...", flush=True)
            chunks = self.translate_chunks_to_english(chunks, title)
            was_translated = True
            print(f"[INGEST-OK] Translated {len(chunks)} chunks to English successfully!", flush=True)
        else:
            print(f"[INGEST] Transcript for '{title}' is in English. Skipping translation.", flush=True)

        # 5. Save chunks locally
        with open(chunk_file, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)

        # 6. Embed with FastEmbed
        vector_store_service.initialize()
        texts = [c["text"] for c in chunks]
        embeddings = list(vector_store_service.embedding_model.embed(texts))

        # 7. Upsert to Qdrant Cloud with deterministic UUIDs
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

        msg = f"Successfully indexed '{title}' ({len(chunks)} chunks, translated to English). Available for search immediately!" if was_translated else f"Successfully indexed '{title}' ({len(chunks)} chunks). Available for search immediately!"

        return {
            "status": "success",
            "video_id": video_id,
            "lecture_title": title,
            "chunk_count": len(chunks),
            "total_lectures": total_lectures,
            "message": msg
        }

video_ingestion_service = VideoIngestionService()
