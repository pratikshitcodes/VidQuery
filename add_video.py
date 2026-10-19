#!/usr/bin/env python3
"""
CLI Tool for On-Demand Incremental Video Ingestion
Usage:
    python add_video.py "https://www.youtube.com/watch?v=VIDEO_ID"
    python add_video.py "https://youtu.be/VIDEO_ID" "DP 32: Custom Lecture Title"
"""
import sys
import os
import argparse

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv

# Ensure environment variables are loaded
load_dotenv()
if not os.getenv("QDRANT_URL"):
    # Try parent directory .env
    parent_env = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), ".env")
    if os.path.exists(parent_env):
        load_dotenv(parent_env)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backend.app.services.video_ingestion import video_ingestion_service

def main():
    parser = argparse.ArgumentParser(description="Add and index a YouTube lecture into Striver DP Reviser RAG.")
    parser.add_argument("url", help="YouTube video URL or 11-character video ID")
    parser.add_argument("--title", "-t", default=None, help="Optional custom lecture title")
    args = parser.parse_args()

    print("=" * 60)
    print("[INGEST] Striver DP Reviser - On-Demand Video Ingestion")
    print(f"Target: {args.url}")
    print("=" * 60)

    try:
        res = video_ingestion_service.ingest_video(url_or_id=args.url, custom_title=args.title)
        print("\n" + "=" * 60)
        print(f"Status: {res['status'].upper()}")
        print(f"Video ID: {res['video_id']}")
        print(f"Title: {res['lecture_title']}")
        print(f"Chunks: {res['chunk_count']}")
        print(f"Total Indexed Lectures: {res['total_lectures']}")
        print(f"Message: {res['message']}")
        print("=" * 60)
    except Exception as e:
        print(f"\n[ERROR] Failed to add video: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
