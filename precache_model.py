# Pre-cache FastEmbed ONNX model during build phase
try:
    print("[BUILD] Pre-caching FastEmbed ONNX model ('sentence-transformers/all-MiniLM-L6-v2')...", flush=True)
    from fastembed import TextEmbedding
    model = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")
    list(model.embed(["warmup"]))
    print("[BUILD] FastEmbed ONNX model cached successfully!", flush=True)
except Exception as e:
    print(f"[BUILD-WARN] Model pre-cache skipped: {e}", flush=True)
