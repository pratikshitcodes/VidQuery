import os
import re
import json
import shutil
import whisper

audio_dir = "data/audio"
raw_transcripts_dir = "data/raw_transcripts"
chunks_dir = "data/chunks"
backup_chunks_dir = "data/chunks_backup_75s"

os.makedirs(raw_transcripts_dir, exist_ok=True)
os.makedirs(chunks_dir, exist_ok=True)

# Exact topic titles for the 31 videos so every chunk carries its lecture context
VIDEO_TITLES = {
    "-zI4mrF2Pb4": "DP 26: Print Longest Common Subsequence (LCS)",
    "3WaxQMELSkw": "DP 6: House Robber II (Circular Street)",
    "6i_T5kkfv4A": "DP 28: Longest Palindromic Subsequence (LPS)",
    "7win3dcgo3k": "DP 15: Partition Equal Subset Sum",
    "AE39gJYuRog": "DP 7: Ninja's Training (2D DP)",
    "EgG3jsGoPvQ": "DP 3: Frog Jump (1 or 2 Stairs)",
    "GS_OqZb2CWc": "DP 16: Partition Set Into 2 Subsets With Minimum Absolute Sum Difference",
    "GqOmJHQZivw": "DP 19: 0/1 Knapsack Problem",
    "GrMBfJNk_NY": "DP 5: Maximum Sum of Non-Adjacent Elements (House Robber I)",
    "HgyouUi11zk": "DP 22: Coin Change 2 (Ways to Make Coin Change)",
    "Kmh3rhyEtB8": "DP 4: Frog Jump with K Distances (Up to K Stairs)",
    "NPZn9jBrX8U": "DP 25: Longest Common Subsequence (LCS)",
    "N_aJ5qQbYA0": "DP 12: Minimum/Maximum Falling Path Sum in Matrix (Variable Start and End)",
    "OgvOZ6OrJoY": "DP 23: Unbounded Knapsack",
    "QGfn7JeXK54": "DP 13: Cherry Pickup II / Chocolates Pickup (3D DP on Grids)",
    "SrP-PiLSYC0": "DP 11: Triangle Minimum Path Sum (Fixed Start, Variable End)",
    "TmhpgXScLyY": "DP 9: Grid Unique Paths II (Maze with Obstacles)",
    "Yz4LlDSlkns": "DP 51: Burst Balloons (Partition DP / MCM)",
    "ZHyb-A2Mte4": "DP 17: Count Subsets with Sum K",
    "_rgTlyky1uQ": "DP 10: Minimum Path Sum in Grid",
    "_wP9mWNPL5w": "DP 27: Longest Common Substring",
    "b3GD8263-PQ": "DP 21: Target Sum",
    "fWX9xDmIzRI": "DP 14: Subset Sum Equal to Target",
    "mLfjzJsN8us": "DP 2: Climbing Stairs",
    "mO8XpGoJwuo": "DP 24: Rod Cutting Problem",
    "myPeWb3Y68A": "DP 20: Minimum Coins (Coin Change)",
    "sdE0A2Oxofw": "DP 8: Grid Unique Paths (Total Unique Paths in Matrix)",
    "tyB0ztf0DNY": "DP 1: Introduction to Dynamic Programming (Fibonacci Memoization & Tabulation)",
    "xPBLEj41rFU": "DP 29: Minimum Insertions to Make a String Palindrome",
    "yMnH0jrir0Q": "DP 30: Minimum Insertions/Deletions to Convert String 1 to String 2",
    "zoilQD1kYSg": "DP 18: Count Partitions With Given Difference D",
}


def split_block_into_fine_segments(block):
    """Splits a 75s chunk block into sentence-level ~5s segments with interpolated timestamps."""
    start = float(block["start"])
    end = float(block["end"])
    text = block["text"].strip()
    # Strip any previously added [Video: ...] prefix if re-running
    text = re.sub(r"^\[Video:.*?\]\s*", "", text)
    if not text:
        return []

    # Split on sentence boundaries (. ? !) or every ~18 words if unpunctuated
    raw_sentences = [s.strip() for s in re.split(r"(?<=[.?!])\s+", text) if s.strip()]
    fine_pieces = []
    for sent in raw_sentences:
        words = sent.split()
        if len(words) > 24:
            for i in range(0, len(words), 18):
                fine_pieces.append(" ".join(words[i : i + 18]))
        else:
            fine_pieces.append(sent)

    total_chars = sum(len(p) for p in fine_pieces) or 1
    duration = max(end - start, 1.0)

    segments = []
    cur_start = start
    for idx, piece in enumerate(fine_pieces):
        frac = len(piece) / total_chars
        cur_end = end if idx == len(fine_pieces) - 1 else round(cur_start + duration * frac, 2)
        segments.append({
            "start": round(cur_start, 2),
            "end": round(cur_end, 2),
            "text": piece
        })
        cur_start = cur_end
    return segments


# ---------------------------------------------------------------------------
# STEP 1: Ensure all existing 31 videos have raw transcripts in data/raw_transcripts/
# ---------------------------------------------------------------------------
if not os.path.exists(backup_chunks_dir) and os.path.exists(chunks_dir):
    shutil.copytree(chunks_dir, backup_chunks_dir)
    print(f"Backed up original 75s chunks to -> {backup_chunks_dir}")

for filename in sorted(os.listdir(chunks_dir)):
    if not filename.endswith(".json"):
        continue
    video_id = os.path.splitext(filename)[0]
    raw_path = os.path.join(raw_transcripts_dir, f"{video_id}.json")

    if not os.path.exists(raw_path):
        src_chunk_path = os.path.join(chunks_dir, filename)
        with open(src_chunk_path, "r", encoding="utf-8") as f:
            old_chunks = json.load(f)

        fine_segments = []
        for blk in old_chunks:
            fine_segments.extend(split_block_into_fine_segments(blk))

        with open(raw_path, "w", encoding="utf-8") as f:
            json.dump(fine_segments, f, ensure_ascii=False, indent=2)
        print(f"Saved raw transcript ({len(fine_segments)} segments) -> {raw_path}")


# ---------------------------------------------------------------------------
# STEP 2: For any NEW audio file in data/audio not yet in data/raw_transcripts,
#         run Whisper ONCE and save its raw segments to data/raw_transcripts/
# ---------------------------------------------------------------------------
whisper_model = None
if os.path.exists(audio_dir):
    for filename in sorted(os.listdir(audio_dir)):
        if not filename.endswith(".webm"):
            continue
        video_id = os.path.splitext(filename)[0]
        raw_path = os.path.join(raw_transcripts_dir, f"{video_id}.json")

        if os.path.exists(raw_path):
            continue  # Already transcribed! Skip Whisper!

        if whisper_model is None:
            whisper_model = whisper.load_model("tiny")

        audio_path = os.path.join(audio_dir, filename)
        print(f"\nTranscribing NEW audio with Whisper: {video_id}")
        result = whisper_model.transcribe(audio_path)
        segments = [
            {
                "start": round(float(seg["start"]), 2),
                "end": round(float(seg["end"]), 2),
                "text": seg["text"].strip(),
            }
            for seg in result["segments"]
        ]
        with open(raw_path, "w", encoding="utf-8") as f:
            json.dump(segments, f, ensure_ascii=False, indent=2)
        print(f"Saved new raw transcript ({len(segments)} segments) -> {raw_path}")


# ---------------------------------------------------------------------------
# STEP 3: Chunking Strategy (Contextual Sliding-Window: 55s Window + 15s Overlap)
# ---------------------------------------------------------------------------
CHUNK_WINDOW_SECONDS = 55.0   # ~150-170 words (fits 100% inside MiniLM's 256-token limit!)
CHUNK_OVERLAP_SECONDS = 15.0  # 15s sliding overlap so no sentence/formula is cut in half!


def build_chunks_from_segments(segments, video_id, window_sec=CHUNK_WINDOW_SECONDS, overlap_sec=CHUNK_OVERLAP_SECONDS):
    if not segments:
        return []

    topic_title = VIDEO_TITLES.get(video_id, f"Video {video_id}")
    chunks = []
    n = len(segments)
    idx = 0

    while idx < n:
        chunk_start = segments[idx]["start"]
        current_segs = []
        j = idx

        while j < n:
            current_segs.append(segments[j])
            if segments[j]["end"] - chunk_start >= window_sec:
                j += 1
                break
            j += 1

        chunk_end = current_segs[-1]["end"]
        raw_body = " ".join(s["text"].strip() for s in current_segs if s["text"].strip())
        # Contextual Header prepended so both Qdrant & Cross-Encoder know the exact lecture topic
        contextual_text = f"[Video: {topic_title}] {raw_body}"

        chunks.append({
            "start": chunk_start,
            "end": chunk_end,
            "text": contextual_text
        })

        if j >= n:
            break

        # Advance pointer with sliding overlap (step back by overlap_sec from chunk_end)
        target_next_start = chunk_end - overlap_sec
        next_idx = idx + 1
        for k in range(idx + 1, j):
            if segments[k]["start"] >= target_next_start:
                next_idx = k
                break
        else:
            next_idx = j

        idx = max(next_idx, idx + 1)

    return chunks


total_chunks_all = 0
for filename in sorted(os.listdir(raw_transcripts_dir)):
    if not filename.endswith(".json"):
        continue
    video_id = os.path.splitext(filename)[0]
    raw_path = os.path.join(raw_transcripts_dir, filename)

    with open(raw_path, "r", encoding="utf-8") as f:
        segments = json.load(f)

    new_chunks = build_chunks_from_segments(segments, video_id)
    output_path = os.path.join(chunks_dir, f"{video_id}.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(new_chunks, f, ensure_ascii=False, indent=2)

    total_chunks_all += len(new_chunks)
    print(f"[{video_id}] {len(segments)} raw segments -> {len(new_chunks)} contextual overlapping chunks")

print(f"\nDone! Generated {total_chunks_all} total chunks across all videos in '{chunks_dir}/'.")