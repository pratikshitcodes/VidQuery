import os
import sys
import json
import time
import re
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer, CrossEncoder
from groq import Groq

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

client = QdrantClient(api_key=QDRANT_API_KEY, url=QDRANT_URL, timeout=60)
COLLECTION_NAME = "knowledge"

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

groq_client = Groq(api_key=GROQ_API_KEY)
LLM_MODEL = "openai/gpt-oss-120b"


def search(query, top_k=5):
    query_embedding = embedding_model.encode(query).tolist()
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        limit=15,
        query=query_embedding,
        with_payload=True
    )
    candidates = results.points
    pairs = [[query, r.payload["text"]] for r in candidates]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [r for r, s in ranked[:top_k]]


def ask_llm(context, query):
    prompt = f"""
    Answer the question using only the context below.

    Context:
    {context}

    Question:
    {query}

    Give a clear and concise answer.

    For each relevant part, mention its timestamp in this format:
    [START - END]

    Do not create or modify any YouTube URLs.

    If the answer is not present in the context, say:
    "I don't have enough information."
    """
    messages = [{"role": "user", "content": prompt}]
    for _ in range(8):
        try:
            res = groq_client.chat.completions.create(model=LLM_MODEL, messages=messages)
            return res.choices[0].message.content
        except Exception as e:
            if "429" in str(e) or "rate_limit" in str(e).lower():
                time.sleep(4.0)
            else:
                raise e
    res = groq_client.chat.completions.create(model=LLM_MODEL, messages=messages)
    return res.choices[0].message.content


def evaluate_with_judge(query, context, generated_answer, ground_truth, expected_info):
    """
    LLM-as-a-Judge: Evaluates the 5 RAG Triad / Ragas Metrics on a 0.0 to 1.0 scale:
    1. Context Precision: Out of retrieved chunks, how many are useful for answering the question.
    2. Context Recall: Does retrieved context contain all key facts present in expected_info / ground truth.
    3. Faithfulness: Are claims in generated answer 100% grounded in context (no hallucination).
    4. Answer Relevance: Does the generated answer directly address the user's question, covering expected_info without fluff.
    5. Answer Correctness: Does the generated answer factually match the ground truth.
    """
    expected_info_str = "\n".join(f"- {item}" for item in expected_info)

    judge_prompt = f"""
    You are an expert impartial AI evaluator evaluating a Retrieval-Augmented Generation (RAG) system.
    Evaluate the response using the RAG Triad / Ragas framework.

    [USER QUERY]:
    {query}

    [GROUND TRUTH REFERENCE]:
    {ground_truth}

    [EXPECTED KEY INFORMATION / FACTS]:
    {expected_info_str}

    [RETRIEVED CONTEXT CHUNKS]:
    {context}

    [GENERATED ANSWER]:
    {generated_answer}

    Score each metric on a continuous scale from 0.0 to 1.0:
    1. "context_precision": Ratio of retrieved context chunks that are directly relevant/useful to the query and expected info (1.0 = highly precise/focused, 0.0 = completely irrelevant).
    2. "context_recall": Fraction of items in [EXPECTED KEY INFORMATION / FACTS] that are successfully present/covered in the retrieved context (1.0 = all expected facts present in context, 0.0 = none).
    3. "faithfulness": Degree to which the generated answer relies ONLY on the retrieved context (1.0 = every statement is backed by context, 0.0 = hallucinated or ungrounded).
    4. "answer_relevance": How directly, clearly, and concisely the answer responds to the query and addresses the expected info without irrelevant fluff (1.0 = perfectly on-topic, 0.0 = completely off-topic or evasive).
    5. "answer_correctness": Factual and semantic alignment between the generated answer and the ground truth reference (1.0 = completely correct, 0.0 = factually wrong).

    Output ONLY a valid JSON object matching this schema:
    {{
        "context_precision": float,
        "context_recall": float,
        "faithfulness": float,
        "answer_relevance": float,
        "answer_correctness": float,
        "reasoning": "1-2 sentence concise explanation"
    }}
    """
    messages = [{"role": "user", "content": judge_prompt}]
    for _ in range(8):
        try:
            res = groq_client.chat.completions.create(
                model=LLM_MODEL,
                messages=messages,
                response_format={"type": "json_object"}
            )
            raw = res.choices[0].message.content
            return json.loads(raw)
        except Exception as e:
            if "429" in str(e) or "rate_limit" in str(e).lower():
                time.sleep(4.0)
            else:
                try:
                    m = re.search(r"\{.*\}", str(e), re.DOTALL)
                    if m:
                        return json.loads(m.group(0))
                except Exception:
                    pass
                time.sleep(2.0)

    return {
        "context_precision": 0.0,
        "context_recall": 0.0,
        "faithfulness": 0.0,
        "answer_relevance": 0.0,
        "answer_correctness": 0.0,
        "reasoning": "Judge evaluation failed"
    }


def run_evaluation():
    golden_path = "data/golden_dataset.json"
    if not os.path.exists(golden_path):
        raise FileNotFoundError(f"Golden dataset not found at {golden_path}")

    with open(golden_path, "r", encoding="utf-8") as f:
        golden_data = json.load(f)

    print("=" * 80)
    print(f"STARTING RAG EVALUATION ON GOLDEN DATASET ({len(golden_data)} QUERIES)")
    print("Model Evaluator: Groq openai/gpt-oss-120b (LLM-as-a-Judge)")
    print("Metrics: Context Precision | Context Recall | Faithfulness | Answer Relevance | Correctness")
    print("=" * 80 + "\n", flush=True)

    eval_results = []

    for item in golden_data:
        q_id = item["id"]
        vid = item.get("lecture_title", item["video_id"])
        query = item["query"]
        gt = item["ground_truth_answer"]
        expected_info = item.get("expected_info", [])

        t0 = time.perf_counter()
        retrieved_points = search(query, top_k=5)
        search_time = (time.perf_counter() - t0) * 1000

        context_str = ""
        for p in retrieved_points:
            pl = p.payload
            context_str += f"[{int(pl['start'])} - {int(pl['end'])}] (video: {pl['video_id']})\n{pl['text']}\n\n"

        t1 = time.perf_counter()
        answer = ask_llm(context_str, query)
        gen_time = (time.perf_counter() - t1) * 1000

        # LLM-as-a-Judge Evaluation with Expected Info & Ground Truth
        scores = evaluate_with_judge(query, context_str, answer, gt, expected_info)

        record = {
            "id": q_id,
            "video": vid,
            "query": query,
            "ground_truth": gt,
            "expected_info": expected_info,
            "generated_answer": answer,
            "context_precision": scores.get("context_precision", 0.0),
            "context_recall": scores.get("context_recall", 0.0),
            "faithfulness": scores.get("faithfulness", 0.0),
            "answer_relevance": scores.get("answer_relevance", 0.0),
            "answer_correctness": scores.get("answer_correctness", 0.0),
            "reasoning": scores.get("reasoning", ""),
            "retrieval_ms": round(search_time, 1),
            "generation_ms": round(gen_time, 1)
        }
        eval_results.append(record)

        print(f"[{q_id}/{len(golden_data)}] {vid}")
        print(f"  Query: {query[:75]}...")
        print(f"  Scores -> Precision: {record['context_precision']:.2f} | Recall: {record['context_recall']:.2f} | Faithfulness: {record['faithfulness']:.2f} | Relevancy: {record['answer_relevance']:.2f} | Correctness: {record['answer_correctness']:.2f}")
        print(f"  Reason: {record['reasoning']}\n", flush=True)

    # Calculate overall averages
    avg_prec = sum(r["context_precision"] for r in eval_results) / len(eval_results)
    avg_rec = sum(r["context_recall"] for r in eval_results) / len(eval_results)
    avg_faith = sum(r["faithfulness"] for r in eval_results) / len(eval_results)
    avg_rel = sum(r["answer_relevance"] for r in eval_results) / len(eval_results)
    avg_corr = sum(r["answer_correctness"] for r in eval_results) / len(eval_results)

    print("=" * 80)
    print(f"FINAL RAG EVALUATION BENCHMARK SUMMARY ({len(golden_data)} QUERIES)")
    print("=" * 80)
    print(f"Context Precision:   {avg_prec:.3f} ({avg_prec * 100:.1f}%)")
    print(f"Context Recall:      {avg_rec:.3f} ({avg_rec * 100:.1f}%)")
    print(f"Faithfulness:        {avg_faith:.3f} ({avg_faith * 100:.1f}%)")
    print(f"Answer Relevance:    {avg_rel:.3f} ({avg_rel * 100:.1f}%)")
    print(f"Answer Correctness:  {avg_corr:.3f} ({avg_corr * 100:.1f}%)")
    print("=" * 80)

    summary_data = {
        "averages": {
            "context_precision": round(avg_prec, 3),
            "context_recall": round(avg_rec, 3),
            "faithfulness": round(avg_faith, 3),
            "answer_relevance": round(avg_rel, 3),
            "answer_correctness": round(avg_corr, 3)
        },
        "records": eval_results
    }

    out_file = "data/rag_eval_results_30q.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    print(f"Detailed evaluation saved to '{out_file}'.")


# Run when invoked
run_evaluation()
