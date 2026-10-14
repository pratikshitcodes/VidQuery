import os
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
from qdrant_client import QdrantClient
from dotenv import load_dotenv
from qdrant_client.models import Distance,VectorParams,PointStruct
from sentence_transformers import SentenceTransformer, CrossEncoder
from groq import Groq
import json
import time
import statistics

load_dotenv()
GROQ_API_KEY=os.getenv("GROQ_API_KEY")
QDRANT_URL=os.getenv("QDRANT_URL")
QDRANT_API_KEY=os.getenv("QDRANT_API_KEY")

client=QdrantClient(
    api_key=QDRANT_API_KEY,
    url=QDRANT_URL,
    timeout=60
)

COLLECTION_NAME="knowledge"
embedding_size=384

REINDEX_QDRANT = False  # Already indexed 1098 contextual chunks! Set to True only after changing main.py chunking

#load data
chunks_dir = "data/chunks"

documents = []

for filename in sorted(os.listdir(chunks_dir)):
    if filename.endswith(".json"):
        file_path = os.path.join(chunks_dir, filename)
        with open(file_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)

        video_id = os.path.splitext(filename)[0]
        for chunk in chunks:
            chunk["video_id"] = video_id

        documents.extend(chunks)

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

if REINDEX_QDRANT:
    print(f"Re-indexing {len(documents)} chunks into Qdrant collection '{COLLECTION_NAME}'...")
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=embedding_size,
            distance=Distance.COSINE
        )
    )

    texts = [doc["text"] for doc in documents]
    embeddings = embedding_model.encode(texts, show_progress_bar=True)

    points = [
        PointStruct(
            id=i + 1,
            vector=embeddings[i].tolist(),
            payload=documents[i]
        )
        for i in range(len(documents))
    ]

    BATCH_SIZE = 250
    for start_idx in range(0, len(points), BATCH_SIZE):
        batch = points[start_idx : start_idx + BATCH_SIZE]
        client.upsert(collection_name=COLLECTION_NAME, points=batch)
    print(f"Successfully indexed {len(points)} chunks into Qdrant!\n")

def search(query, top_k=5):
    t0 = time.perf_counter()
    # Step 1: Qdrant se more candidates retrieve karo
    query_embedding = embedding_model.encode(query).tolist()

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        limit=15,
        query=query_embedding,
        with_payload=True
    )
    qdrant_ms = (time.perf_counter() - t0) * 1000

    candidates = results.points

    # Step 2: Query + chunk pairs banao
    t1 = time.perf_counter()
    pairs = [
        [query, result.payload["text"]]
        for result in candidates
    ]

    # Step 3: Reranker scores calculate karega
    scores = reranker.predict(pairs)

    # Step 4: Score ke basis par sort karo
    ranked = sorted(
        zip(candidates, scores),
        key=lambda x: x[1],
        reverse=True
    )
    rerank_ms = (time.perf_counter() - t1) * 1000

    # Step 5: Best top_k return karo
    return [result for result, score in ranked[:top_k]], qdrant_ms, rerank_ms

queries = [
    # 1. -zI4mrF2Pb4.json (DP 26: Print Longest Common Subsequence)
    # [EXISTING 1]
    "What is the length and the actual string of the Longest Common Subsequence (LCS) for strings EBCDE and BDGEK as demonstrated in the video?",
    # [EXISTING 2]
    "When backtracking through the 2D DP table to reconstruct the LCS string, what action is taken when characters match (S1[i-1] == S2[j-1]), and where do the pointers move?",
    # [NEW 1]
    "In the Print LCS backtracking step, if the characters do not match and both adjacent DP cells (dp[i-1][j] and dp[i][j-1]) have the exact same value, which direction can the pointer move?",

    # 2. 3WaxQMELSkw.json (DP 6: House Robber II)
    # [EXISTING 3]
    "What is the key constraint difference between House Robber I (Lecture 5) and House Robber II (Lecture 6)?",
    # [EXISTING 4]
    "How is the House Robber II problem decomposed into two subproblems to reuse the House Robber I logic?",
    # [NEW 2]
    "When implementing House Robber II in code using the Lecture 5 space-optimized function, what two temporary arrays are created from the original array and what edge case is checked if the array has only 1 element?",

    # 3. 6i_T5kkfv4A.json (DP 28: Longest Palindromic Subsequence)
    # [EXISTING 5]
    "How does the author reduce the Longest Palindromic Subsequence (LPS) problem into the Longest Common Subsequence (LCS) problem?",
    # [EXISTING 6]
    "What are the two brute-force methods mentioned to generate all subsequences of a string before optimizing with DP?",
    # [NEW 3]
    "In the Longest Palindromic Subsequence lecture, what example string is reversed to demonstrate how the longest common subsequence between the string and its reverse forms a palindrome?",

    # 4. 7win3dcgo3k.json (DP 15: Partition Equal Subset Sum)
    # [EXISTING 7]
    "What is the first condition checked regarding the total sum S of the array in Partition Equal Subset Sum, and what happens if S is odd?",
    # [EXISTING 8]
    "What edge case correction from a viewer comment was addressed regarding array[0] when initializing the DP table of size k + 1?",
    # [NEW 4]
    "In Partition Equal Subset Sum, why is finding just one subset with sum equal to S / 2 sufficient to guarantee that the remaining elements also sum to S / 2?",

    # 5. AE39gJYuRog.json (DP 7: Ninja's Training)
    # [EXISTING 9]
    "In the Ninja's Training problem, why does a greedy approach fail on the example where day 0 has options [10, 50, ...] and day 1 has [something, 100, ...]?",
    # [EXISTING 10]
    "What is the time complexity and auxiliary space complexity of the memoized recursive solution for Ninja's Training?",
    # [NEW 5]
    "In the tabulation base case for Ninja's Training (day 0), how are dp[0][0], dp[0][1], dp[0][2], and dp[0][3] initialized?",

    # 6. EgG3jsGoPvQ.json (DP 3: Frog Jump)
    # [EXISTING 11]
    "In Frog Jump (Lecture 3), what are the two possible energy costs the frog can incur when jumping from stair i?",
    # [EXISTING 12]
    "How is the space complexity optimized from O(N) in tabulation down to O(1) in the Frog Jump problem?",
    # [NEW 6]
    "In the Frog Jump recursion tree dry run, how does memoization avoid recomputing overlapping subproblems like f(1) or f(2), and what is the resulting time complexity?",

    # 7. GS_OqZb2CWc.json (DP 16: Partition Set Into 2 Subsets With Min Absolute Sum Diff)
    # [EXISTING 13]
    "Which specific row of the DP table contains all possible subset sums that can be formed using all n elements of the array?",
    # [EXISTING 14]
    "What formula is used to calculate the absolute difference between the two subset sums if one subset has sum s1?",
    # [NEW 7]
    "In DP 16 (Partition Set Into 2 Subsets With Minimum Absolute Sum Difference), how does the Subset Sum tabulation table help check multiple target sums from 0 to total_sum without re-running recursion for each target?",

    # 8. GqOmJHQZivw.json (DP 19: 0/1 Knapsack)
    # [EXISTING 15]
    "Why does a greedy approach based on picking the highest value item first fail for the 0/1 Knapsack problem?",
    # [EXISTING 16]
    "When optimizing the space complexity of 0/1 Knapsack to a single 1D array, why must the capacity loop run from W down to 0 instead of 0 up to W?",
    # [NEW 8]
    "In the 0/1 Knapsack tabulation base case at index 0, for which values of capacity W can the thief steal the item at index 0 (wt[0])?",

    # 9. GrMBfJNk_NY.json (DP 5: Maximum Sum of Non-Adjacent Elements / House Robber I)
    # [EXISTING 17]
    "What is the time complexity of the DP solution for Maximum Sum of Non-Adjacent Elements (House Robber I)?",
    # [NEW 9]
    "In Maximum Sum of Non-Adjacent Elements (Lecture 5), if you pick the element at the current index, why must the recursive call jump to index - 2 instead of index - 1?",
    # [NEW 10]
    "In the tabulation approach for Maximum Sum of Non-Adjacent Elements (Lecture 5), what edge case occurs when computing index - 2 for i = 1, and how is it handled?",

    # 10. HgyouUi11zk.json (DP 22: Coin Change 2 / Ways to Make Coin Change)
    # [EXISTING 18]
    "In the pick condition for Coin Change II, why does the index not decrement after choosing coin T[ind]?",
    # [EXISTING 19]
    "What does the base case return at index 0 if the remaining target value T is divisible by coins[0] in Coin Change 2?",
    # [NEW 11]
    "In Coin Change 2 (Ways to Make Coin Change), why is the recursive time complexity exponential and even worse than O(2^N)?",

    # 11. Kmh3rhyEtB8.json (DP 4: Frog Jump with K Distances)
    # [EXISTING 20]
    "How does the recurrence relation for Frog Jump generalize when the frog is allowed to jump up to k stairs instead of just 1 or 2 stairs?",
    # [EXISTING 21]
    "Can the space complexity of Frog Jump with K Distances be optimized to O(1) when k equals N?",
    # [NEW 12]
    "In Frog Jump with K Distances (Lecture 4), why is an inner loop from j = 1 to k needed inside the recursion or tabulation, and what boundary condition must be checked for index - j?",

    # 12. NPZn9jBrX8U.json (DP 25: Longest Common Subsequence Introduction)
    # [EXISTING 22]
    "What is the definition of a subsequence in a string, and how does it differ from a substring?",
    # [EXISTING 23]
    "In the recursive formulation of LCS, how do the indices move when characters at S1[i] and S2[j] do not match?",
    # [NEW 13]
    "When converting the Longest Common Subsequence (LCS) recursion to tabulation, why do we shift the indices by +1 (1-based indexing) in the DP table?",

    # 13. N_aJ5qQbYA0.json (DP 12: Minimum/Maximum Falling Path Sum in Matrix)
    # [EXISTING 24]
    "What are the three possible moves a person can make from cell (i, j) in the variable starting/ending point matrix problem?",
    # [EXISTING 25]
    "What value is returned in the recursion when a path moves out of matrix column boundaries (j < 0 or j >= m)?",
    # [NEW 14]
    "In the space-optimized solution for Maximum Falling Path Sum (DP 12), how is the final maximum answer obtained after computing the last row?",

    # 14. OgvOZ6OrJoY.json (DP 23: Unbounded Knapsack)
    # [EXISTING 26]
    "How does Unbounded Knapsack differ fundamentally from the 0/1 Knapsack problem in terms of item supply?",
    # [EXISTING 27]
    "In Unbounded Knapsack, what does the base case return at index 0 when the remaining capacity is W and the item weight is wt[0]?",
    # [NEW 15]
    "In Unbounded Knapsack space optimization to a single 1D array, why can we overwrite the same row from left to right using only one array instead of two?",

    # 15. QGfn7JeXK54.json (DP 13: Chocolates Pickup / Cherry Pickup II - 3D DP)
    # [EXISTING 28]
    "Where do Alice and Bob start their journeys in the grid, and what row-by-row constraint do they follow?",
    # [EXISTING 29]
    "How are chocolates counted when both Alice and Bob land on the exact same cell in the grid?",
    # [NEW 16]
    "In Cherry Pickup II (3D DP), since both Alice and Bob can each move in 3 directions (-1, 0, +1), how many total combinations of simultaneous moves are explored at each step using nested loops?",

    # 16. SrP-PiLSYC0.json (DP 11: Triangle Minimum Path Sum)
    # [EXISTING 30]
    "Why does the Triangular Grid problem have a fixed starting point but variable ending points?",
    # [EXISTING 31]
    "Why is starting recursion from (0, 0) and moving downward more natural for the triangle problem than starting from the bottom row?",
    # [NEW 17]
    "In the tabulation method for the Triangle Minimum Path Sum problem, why do the loops run in the opposite direction of the top-down recursion (from bottom row n - 2 up to 0)?",

    # 17. TmhpgXScLyY.json (DP 9: Grid Unique Paths II / Maze Obstacles)
    # [EXISTING 32]
    "How is a cell with an obstacle represented in the grid, and what should dp[i][j] be set to when an obstacle is encountered?",
    # [EXISTING 33]
    "What is the base case value when reaching cell (0, 0) assuming there is no obstacle at (0, 0)?",
    # [NEW 18]
    "In Grid Unique Paths II (Maze Obstacles), why is a modulo operation performed before storing values in the DP table during the coding walkthrough?",

    # 18. Yz4LlDSlkns.json (DP 51: Burst Balloons)
    # [EXISTING 34]
    "Why does solving subproblems independently fail if you choose which balloon to burst first?",
    # [EXISTING 35]
    "Why does thinking in reverse (choosing which balloon is bursted LAST) make the subproblems independent?",
    # [NEW 19]
    "In the Burst Balloons problem, when balloon index ind is chosen as the last balloon to burst in the range [i, j], what is the formula for the coins gained from bursting ind itself?",

    # 19. ZHyb-A2Mte4.json (DP 17: Count Subsets with Sum K)
    # [EXISTING 36]
    "Why does the existence of zeros (0) in the array require special handling in Count Subsets with Sum K?",
    # [EXISTING 37]
    "What does the recurrence relation add together at each step in Count Subsets with Sum K: the pick result and what other result?",
    # [NEW 20]
    "In Count Subsets with Sum K (DP 17), what are the two state variables in the DP table and what are the dimensions of the 2D DP array?",

    # 20. _rgTlyky1uQ.json (DP 10: Minimum Path Sum in Grid)
    # [EXISTING 38]
    "What two directions of movement are permitted in the Minimum Path Sum in Grid problem?",
    # [NEW 21]
    "In Minimum Path Sum in Grid (DP 10), when moving backward from (m - 1, n - 1) to (0, 0) in recursion, what are the two recursive calls made at cell (i, j) before taking their minimum?",
    # [NEW 22]
    "In Minimum Path Sum in Grid (DP 10), why do we add grid[i][j] to both the 'up' and 'left' recursive calls, and why is a large value like 1e9 returned when going out of bounds?",

    # 21. _wP9mWNPL5w.json (DP 27: Longest Common Substring)
    # [EXISTING 39]
    "How does the recurrence for Longest Common Substring differ from LCS when characters do not match (S1[i-1] != S2[j-1])?",
    # [EXISTING 40]
    "Where is the final answer stored in the DP table for Longest Common Substring compared to LCS?",
    # [NEW 23]
    "In Longest Common Substring (DP 27), when filling the 2D DP table for matching characters S1[i-1] == S2[j-1], why do we add 1 to the diagonal value dp[i-1][j-1] and maintain an 'ans' variable?",

    # 22. b3GD8263-PQ.json (DP 21: Target Sum)
    # [EXISTING 41]
    "How is the Target Sum problem mathematically transformed into Partition Equal Subset Sum / Count Subsets with Difference D?",
    # [EXISTING 42]
    "If total_sum - target is negative or is an odd number, what should the Target Sum function return?",
    # [NEW 24]
    "In the Target Sum lecture (DP 21), which previously solved lecture's code (DP 18) is directly copied and reused to solve the problem?",

    # 23. fWX9xDmIzRI.json (DP 14: Subset Sum Equal to Target)
    # [EXISTING 43]
    "What are the two choices considered at each element in the recursive function for Subset Sum Equal to Target?",
    # [EXISTING 44]
    "Why does the condition arr[ind] <= target need to be checked before taking the 'pick' recursive branch in Subset Sum Equal to Target?",
    # [NEW 25]
    "When memoizing Subset Sum Equal to Target (DP 14), why is the DP array initialized with -1 instead of boolean false for unvisited states?",

    # 24. mLfjzJsN8us.json (DP 2: Climbing Stairs)
    # [EXISTING 45]
    "Why is the Climbing Stairs problem equivalent to finding the N-th Fibonacci number?",
    # [NEW 26]
    "What are the three shortcut steps taught in Lecture 2 (Climbing Stairs) for writing any Dynamic Programming recurrence relation in terms of an index?",
    # [NEW 27]
    "In Climbing Stairs (Lecture 2), if the constraint for N is as large as 10^18 in CodeStudio, why does an O(N) loop give Time Limit Exceeded and what technique is mentioned to solve it in O(log N)?",

    # 25. mO8XpGoJwuo.json (DP 24: Rod Cutting Problem)
    # [NEW 28]
    "How is the Rod Cutting problem mapped to the Unbounded Knapsack pattern in terms of trying to collect rod lengths that sum to N to maximize price?",
    # [NEW 29]
    "In the Rod Cutting problem base case at index 0 (where rod piece length is 1), why is the return value equal to N * price[0]?",
    # [NEW 30]
    "In the Rod Cutting problem (DP 24), why is the rod length at a 0-based index 'ind' equal to ind + 1, and what condition must hold before taking that piece?",

    # 26. myPeWb3Y68A.json (DP 20: Minimum Coins / Coin Change)
    # [NEW 31]
    "In Minimum Coins (DP 20), why is 1e9 returned instead of INT_MAX in the base case when a target cannot be formed, specifically when adding 1 + f(ind, T - coins[ind])?",
    # [NEW 32]
    "In Minimum Coins (DP 20), what does the base case at index 0 return if the remaining target T is divisible by coins[0], and what does it return if it is not divisible?",
    # [NEW 33]
    "When converting the Minimum Coins recursion into bottom-up tabulation, what ranges do the index loop and the target loop iterate over?",

    # 27. sdE0A2Oxofw.json (DP 8: Grid Unique Paths)
    # [NEW 34]
    "In Grid Unique Paths (DP 8), why does top-down recursion start from (m - 1, n - 1) and move using 'up' (i - 1, j) and 'left' (i, j - 1) toward (0, 0)?",
    # [NEW 35]
    "In the tabulation solution for Grid Unique Paths (DP 8), how are the 'up' and 'left' variables computed when i > 0 and j > 0?",
    # [NEW 36]
    "When space-optimizing Grid Unique Paths (DP 8) from a 2D DP table to 1D arrays, what do the 'prev' (previous row) and 'temp' / 'cur' (current row) arrays represent?",

    # 28. tyB0ztf0DNY.json (DP 1: Introduction to Dynamic Programming / Fibonacci)
    # [NEW 37]
    "In Lecture 1 (Introduction to Dynamic Programming), what are the two main DP approaches (Memoization and Tabulation) and how do their directions (top-down vs bottom-up) differ?",
    # [NEW 38]
    "In Lecture 1 (Fibonacci), what are the three steps to convert a recursive solution into a memoized solution using a dp array initialized to -1?",
    # [NEW 39]
    "In Lecture 1 (Fibonacci), how is the space complexity reduced from O(N) in tabulation to O(1) using only 'prev' and 'prev2' variables?",

    # 29. xPBLEj41rFU.json (DP 29: Minimum Insertions to Make String Palindrome)
    # [NEW 40]
    "In Minimum Insertions to Make a String Palindrome (DP 29), why is the maximum number of insertions ever needed for a string of length n at most n (by attaching the reverse of the string)?",
    # [NEW 41]
    "In Minimum Insertions to Make a String Palindrome (DP 29), how is the example string 'abcaa' or 'codingninjas' used to show that keeping the Longest Palindromic Subsequence intact minimizes insertions?",
    # [NEW 42]
    "What is the exact formula for the minimum insertions to make a string palindrome in terms of string length n and the length of the Longest Palindromic Subsequence (LPS)?",

    # 30. yMnH0jrir0Q.json (DP 30: Minimum Insertions/Deletions to Convert String 1 to String 2)
    # [NEW 43]
    "In DP 30 (Minimum Operations to Convert String 1 to String 2), using the example strings 'abcd' and 'anc', which common subsequence is kept intact and how many deletions and insertions are performed?",
    # [NEW 44]
    "In DP 30, what are the formulas for the number of deletions (n - len(LCS)) and the number of insertions (m - len(LCS)) to convert String 1 of length n into String 2 of length m?",

    # 31. zoilQD1kYSg.json (DP 18: Count Partitions With Given Difference D)
    # [NEW 45]
    "In Count Partitions With Given Difference D (DP 18), how do the two equations S1 - S2 = D and S1 + S2 = total_sum combine to give S2 = (total_sum - D) / 2?"
]
llm_model="openai/gpt-oss-120b"
groq_client=Groq(
    api_key=GROQ_API_KEY
)
def ask_llm(context,query):
    prompt=f"""
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
    messages=[{
        "role":"user",
        "content":prompt
    }]
    for attempt in range(8):
        try:
            response=groq_client.chat.completions.create(model=llm_model,messages=messages)
            return response.choices[0].message.content
        except Exception as e:
            if "429" in str(e) or "rate_limit" in str(e).lower():
                time.sleep(4.0)
            else:
                raise e
    response=groq_client.chat.completions.create(model=llm_model,messages=messages)
    return response.choices[0].message.content



def format_time(seconds):

    minutes = int(seconds // 60)
    seconds = int(seconds % 60)

    return f"{minutes:02d}:{seconds:02d}"


for idx, query in enumerate(queries, 1):
    results, qdrant_ms, rerank_ms = search(query, top_k=5)
    context = ""
    for point in results:
        payload = point.payload
        start = int(payload["start"])
        end = int(payload["end"])
        text = payload["text"]
        video_id = payload["video_id"]
        context += f"[{start} - {end}] (video: {video_id})\n{text}\n\n"

    t_llm = time.perf_counter()
    answer = ask_llm(context, query)
    llm_ms = (time.perf_counter() - t_llm) * 1000
    total_ms = qdrant_ms + rerank_ms + llm_ms

    print(f"\n[{idx}/90] Query: {query}", flush=True)
    print(f"⏱️ Latency -> Qdrant: {qdrant_ms:.1f}ms | Rerank: {rerank_ms:.1f}ms | LLM: {llm_ms:.1f}ms | Total: {total_ms:.1f}ms", flush=True)
    print(f"\nANSWER:\n{answer}\n", flush=True)

