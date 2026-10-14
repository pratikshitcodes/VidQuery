import os
import streamlit as st
import requests

# Configure page
st.set_page_config(
    page_title="Striver DP Video RAG Reviser",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Backend URL (configurable for Docker via environment variable)
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# Custom CSS for dark modern theme and glowing cards
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
        background: linear-gradient(90deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sub-header {
        color: #94a3b8;
        font-size: 1.0rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 10px 16px;
        margin-bottom: 14px;
        display: flex;
        justify-content: space-around;
        text-align: center;
    }
    .metric-item {
        font-size: 0.85rem;
        color: #cbd5e1;
    }
    .metric-value {
        font-size: 1.1rem;
        font-weight: 700;
        color: #38bdf8;
    }
    .source-card {
        background-color: #0f172a;
        border-left: 4px solid #6366f1;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin-bottom: 12px;
        border-top: 1px solid #1e293b;
        border-right: 1px solid #1e293b;
        border-bottom: 1px solid #1e293b;
    }
    .source-title {
        font-weight: 600;
        color: #e2e8f0;
        font-size: 0.95rem;
    }
    .source-meta {
        font-size: 0.8rem;
        color: #94a3b8;
        margin-bottom: 6px;
    }
    .source-text {
        font-size: 0.88rem;
        color: #cbd5e1;
        line-height: 1.45;
    }
    .stVideo {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid #334155;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5);
    }
</style>
""", unsafe_allow_html=True)

# Helper function
def format_seconds(seconds: int) -> str:
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"

# Check backend health
@st.cache_data(ttl=15)
def check_backend_health():
    try:
        r = requests.get(f"{BACKEND_URL}/health", timeout=3)
        if r.status_code == 200:
            return r.json()
        return None
    except Exception:
        return None

# Fetch all lectures
@st.cache_data(ttl=60)
def fetch_lectures():
    try:
        r = requests.get(f"{BACKEND_URL}/api/v1/lectures", timeout=5)
        if r.status_code == 200:
            return r.json().get("lectures", [])
        return []
    except Exception:
        return []

# Initialize session state variables
if "active_video_id" not in st.session_state:
    st.session_state.active_video_id = "myPeWb3Y68A"
if "active_start_seconds" not in st.session_state:
    st.session_state.active_start_seconds = 969
if "active_lecture_title" not in st.session_state:
    st.session_state.active_lecture_title = "DP 20: Minimum Coins (Coin Change)"
if "last_response" not in st.session_state:
    st.session_state.last_response = None
if "current_query" not in st.session_state:
    st.session_state.current_query = ""

# Sidebar
with st.sidebar:
    st.image("https://img.youtube.com/vi/myPeWb3Y68A/maxresdefault.jpg", use_container_width=True)
    st.title("Settings & Status")
    
    health_info = check_backend_health()
    if health_info and health_info.get("status") == "healthy":
        st.success("🟢 Backend Connected")
        st.caption(f"Qdrant: **{health_info.get('collection_name')}**")
    else:
        st.error(f"🔴 Backend Offline ({BACKEND_URL})")
        st.caption("Ensure FastAPI is running: `uvicorn backend.app.main:app --reload`")
    
    st.markdown("---")
    top_k_val = st.slider("Context Chunks (top_k)", min_value=1, max_value=8, value=5)
    
    st.markdown("---")
    st.markdown("### 📚 Indexed Corpus")
    lectures = fetch_lectures()
    st.caption(f"**31 Striver DP Lectures** | **1,098 Chunks**")
    with st.expander(f"View All {len(lectures)} Lectures"):
        for lec in lectures:
            st.markdown(f"- [{lec['lecture_title']}]({lec['youtube_url']}) `({lec['chunk_count']} chunks)`")

# Top Header
st.markdown('<div class="main-header">🎥 Striver DP Video RAG Reviser</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Millisecond-accurate video grounding with Cross-Encoder re-ranking across Striver\'s Dynamic Programming series.</div>', unsafe_allow_html=True)

# Sample query quick chips
st.markdown("**💡 Try sample questions:**")
sample_queries = [
    "Why is 1e9 returned instead of INT_MAX in Minimum Coins?",
    "How is House Robber II decomposed into two subproblems?",
    "How does the Frog Jump recurrence generalize for up to k stairs?",
    "How does the author reduce LPS to LCS?",
    "Why does the greedy approach fail for 0/1 Knapsack?"
]

chip_cols = st.columns(len(sample_queries))
selected_chip = None
for i, sample in enumerate(sample_queries):
    with chip_cols[i]:
        short_label = sample.split("?")[0][:28] + "..."
        if st.button(short_label, key=f"chip_{i}", use_container_width=True):
            selected_chip = sample

# Main input box
query_input = st.text_input(
    "Ask any technical question about Dynamic Programming:",
    value=selected_chip if selected_chip else st.session_state.current_query,
    placeholder="e.g. In Ninja's Training, what is the time complexity of the memoized solution?"
)

# Submit button
search_clicked = st.button("🔍 Search & Answer", type="primary", use_container_width=False)

if (search_clicked or selected_chip) and query_input.strip():
    st.session_state.current_query = query_input.strip()
    with st.spinner("Searching Qdrant, re-ranking with Cross-Encoder, and generating answer via Groq..."):
        try:
            payload = {"query": query_input.strip(), "top_k": top_k_val}
            res = requests.post(f"{BACKEND_URL}/api/v1/query", json=payload, timeout=30)
            if res.status_code == 200:
                st.session_state.last_response = res.json()
                # Default active video to top source
                if st.session_state.last_response.get("sources"):
                    top_source = st.session_state.last_response["sources"][0]
                    st.session_state.active_video_id = top_source["video_id"]
                    st.session_state.active_start_seconds = top_source["start_seconds"]
                    st.session_state.active_lecture_title = top_source["lecture_title"]
            else:
                st.error(f"Error from backend ({res.status_code}): {res.text}")
        except Exception as e:
            st.error(f"Failed to connect to backend: {str(e)}")

# Two-column synchronized layout
col_left, col_right = st.columns([1.1, 0.9], gap="large")

with col_left:
    if st.session_state.last_response:
        resp = st.session_state.last_response
        latency = resp.get("latency", {})
        
        # Latency metric pills
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-item">⚡ Total E2E<br><span class="metric-value">{latency.get('total_ms', 0):.0f} ms</span></div>
            <div class="metric-item">🔍 Qdrant Search<br><span class="metric-value">{latency.get('qdrant_ms', 0):.0f} ms</span></div>
            <div class="metric-item">🎯 Re-Ranker<br><span class="metric-value">{latency.get('rerank_ms', 0):.0f} ms</span></div>
            <div class="metric-item">🤖 Groq 120B<br><span class="metric-value">{latency.get('llm_ms', 0):.0f} ms</span></div>
        </div>
        """, unsafe_allow_html=True)

        # Generated Answer Box
        st.markdown("### 📝 Synthesized Answer")
        st.markdown(resp.get("answer", ""))

        st.markdown("---")
        st.markdown("### 🔍 Retrieved Context & Speech Excerpts")
        st.caption("Click **'▶️ Play Clip'** on any source card to jump the video player to that exact lecture timestamp!")

        sources = resp.get("sources", [])
        for src in sources:
            rank = src.get("rank")
            title = src.get("lecture_title")
            ts_label = src.get("timestamp_label")
            start_s = src.get("start_seconds")
            score = src.get("rerank_score")
            text = src.get("text")
            vid = src.get("video_id")

            with st.container():
                st.markdown(f"""
                <div class="source-card">
                    <div class="source-title">#{rank} • {title}</div>
                    <div class="source-meta">⏱️ Timestamp: <b>{ts_label}</b> | 🎯 Relevance Score: <b>{score:.3f}</b></div>
                    <div class="source-text">{text}</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Interactive button to sync video player
                btn_cols = st.columns([0.45, 0.55])
                with btn_cols[0]:
                    if st.button(f"▶️ Play Clip ({ts_label})", key=f"play_src_{rank}"):
                        st.session_state.active_video_id = vid
                        st.session_state.active_start_seconds = start_s
                        st.session_state.active_lecture_title = title
                        st.rerun()
                with btn_cols[1]:
                    st.markdown(f"[🔗 Open in YouTube]({src.get('youtube_url')})", unsafe_allow_html=True)
                st.markdown("<div style='margin-bottom: 8px;'></div>", unsafe_allow_html=True)

    else:
        st.info("👋 Enter a query above or click any sample question to see the RAG retrieval in action!")

with col_right:
    st.markdown("### 📺 Synchronized Video Player")
    st.markdown(f"**Lecture:** `{st.session_state.active_lecture_title}`")
    st.markdown(f"**Current Timestamp:** `{format_seconds(st.session_state.active_start_seconds)}` ({st.session_state.active_start_seconds}s)")
    
    # YouTube video embed starting at exact second
    yt_watch_url = f"https://www.youtube.com/watch?v={st.session_state.active_video_id}"
    st.video(yt_watch_url, start_time=st.session_state.active_start_seconds)
    
    st.caption(f"Direct Link: [{yt_watch_url}&t={st.session_state.active_start_seconds}s]({yt_watch_url}&t={st.session_state.active_start_seconds}s)")
    
    st.markdown("---")
    st.markdown("#### 🎯 How the Video Grounding Works")
    st.markdown("""
    1. **Sliding-Window Chunks (55s / 15s Overlap):** Speech audio was transcribed via Whisper into timestamped segments.
    2. **Two-Stage Re-Ranking:** Qdrant finds top-15 semantic candidates, then a Cross-Encoder (`bge-reranker-base`) selects the top-5 most relevant speech slices.
    3. **Zero-Latency Seeking:** Clicking any clip automatically passes the `start_time` parameter directly into the player without reloading the app.
    """)
