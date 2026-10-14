import os
import sys

# Pre-download and cache model weights during the build phase
# so runtime queries never spend time downloading over the network.
try:
    import torch
    torch.set_num_threads(1)
    from sentence_transformers import SentenceTransformer
    print("[BUILD] Pre-caching SentenceTransformer('all-MiniLM-L6-v2')...", flush=True)
    SentenceTransformer("all-MiniLM-L6-v2")
    print("[BUILD] Model successfully cached into local container layer!", flush=True)
except Exception as e:
    print(f"[BUILD-WARN] Model pre-cache skipped: {e}", flush=True)
