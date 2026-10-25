import time
from typing import Tuple
from groq import Groq
from backend.app.config import settings

class LLMService:
    def __init__(self):
        self.client: Groq = None

    def initialize(self):
        """Initializes the Groq API client."""
        if self.client is None:
            self.client = Groq(api_key=settings.GROQ_API_KEY)

    def generate_answer(self, context: str, query: str) -> Tuple[str, float]:
        """
        Sends formatted context and query to Groq LLM with strict negative constraints.
        Returns generated answer and latency in milliseconds.
        """
        if self.client is None:
            self.initialize()

        prompt = f"""You are an expert AI tutor teaching Dynamic Programming based on Striver's A-Z DSA Course video lectures.

Context from video transcripts:
{context}

Question:
{query}

Instructions:
1. Give a clear, technically precise, and concise answer directly addressing the question.
2. Formulate recurrence relations, base cases, and time/space complexities clearly using clean, human-readable notation.
   - Use plain, clean code or mathematical notation (e.g., `O(N)`, `T(N) = 2*T(N/2) + O(1)`, `1e9`, `dp[i][j] = dp[i-1][j] + ...`).
   - Do NOT output raw, messy LaTeX blocks or LaTeX environments such as \\begin{{cases}}, \\text{{...}}, \\frac{{...}}, or unrendered backslash symbols. Keep everything clean, legible, and directly readable.
3. For each key claim or formula, cite the corresponding timestamp in this exact format: [START - END] (e.g. [16:09 - 17:06] or [969 - 1026]).
4. Answer strictly from the provided context. If the context does not contain enough information to answer the question, state:
"The provided lecture excerpts do not contain enough information to answer this question."
Do NOT invent or hallucinate facts beyond the transcripts.
"""
        messages = [
            {"role": "user", "content": prompt}
        ]

        t0 = time.perf_counter()
        
        # Retry with exponential backoff for rate limits
        for attempt in range(6):
            try:
                response = self.client.chat.completions.create(
                    model=settings.LLM_MODEL_NAME,
                    messages=messages,
                    temperature=0.1
                )
                answer = response.choices[0].message.content
                llm_ms = (time.perf_counter() - t0) * 1000.0
                return answer, llm_ms
            except Exception as e:
                err_msg = str(e).lower()
                if "429" in err_msg or "rate_limit" in err_msg:
                    sleep_time = 3.5 * (1.5 ** attempt)
                    time.sleep(sleep_time)
                else:
                    raise e

        # Final attempt
        response = self.client.chat.completions.create(
            model=settings.LLM_MODEL_NAME,
            messages=messages,
            temperature=0.1
        )
        answer = response.choices[0].message.content
        llm_ms = (time.perf_counter() - t0) * 1000.0
        return answer, llm_ms

llm_service = LLMService()
