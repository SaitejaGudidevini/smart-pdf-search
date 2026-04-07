"""RAGAS Evaluation for EDMS RAG Pipeline.

Follows the official RAGAS quickstart pattern:
https://docs.ragas.io/en/stable/getstarted/quickstart/

Usage:
  cd /Users/saiteja/Documents/Dev/EDMS
  source src/smart-search/.env
  RAG_PGHOST=localhost src/smart-search/.venv/bin/python scripts/eval_ragas.py
"""

import json
import os
import sys
import time
import httpx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "smart-search"))

# Load .env from smart-search folder
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "src", "smart-search", ".env"))


def call_groq(system: str, user: str, temperature: float = 0) -> str | None:
    """Call Groq LLM."""
    key = os.environ.get("GROQ_API_KEY", "")
    if not key:
        return None
    try:
        resp = httpx.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "max_tokens": 500, "temperature": temperature,
            },
            timeout=30,
        )
        if resp.status_code == 200:
            return resp.json()["choices"][0]["message"]["content"]
        elif resp.status_code == 429:
            time.sleep(5)
            return call_groq(system, user, temperature)
    except Exception as e:
        print(f"  Groq error: {e}")
    return None


# ══════════════════════════════════════════════════════════
#  STEP 1: Define our RAG system (same as RAGAS quickstart)
# ══════════════════════════════════════════════════════════

import psycopg
from psycopg.rows import dict_row
import httpx
from fastembed import TextEmbedding


class EDMSRAG:
    """Our RAG system — wraps pgvector search + Groq LLM."""

    def __init__(self):
        self.model = TextEmbedding("BAAI/bge-base-en-v1.5")
        self.conn = psycopg.connect(
            host=os.environ.get("RAG_PGHOST", "localhost"),
            port=int(os.environ.get("RAG_PGPORT", "5432")),
            dbname=os.environ.get("RAG_PGDATABASE", "mayan"),
            user=os.environ.get("RAG_PGUSER", "mayan"),
            password=os.environ.get("RAG_PGPASSWORD", "mayan"),
            row_factory=dict_row,
        )
        self.groq_key = os.environ.get("GROQ_API_KEY", "")

    def get_most_relevant_docs(self, query, k=5):
        """Retrieve top-k chunks from pgvector."""
        query_emb = list(self.model.embed([query]))[0].tolist()
        results = self.conn.execute("""
            SELECT content FROM rag.chunks
            WHERE chunk_type = 'child' AND embedding IS NOT NULL
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """, (query_emb, k)).fetchall()
        return [r["content"][:500] for r in results]

    def generate_answer(self, query, relevant_docs):
        """Generate answer using Groq LLM."""
        context = "\n\n".join(f"[Source {i+1}]: {doc}" for i, doc in enumerate(relevant_docs[:3]))
        resp = httpx.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {self.groq_key}", "Content-Type": "application/json"},
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": [
                    {"role": "system", "content": "Answer the question using ONLY the provided sources. Cite source numbers."},
                    {"role": "user", "content": f"Question: {query}\n\n{context}"},
                ],
                "max_tokens": 300, "temperature": 0,
            },
            timeout=30,
        )
        if resp.status_code == 200:
            return resp.json()["choices"][0]["message"]["content"]
        return "Error generating answer"


# ══════════════════════════════════════════════════════════
#  STEP 2: Define sample queries and expected answers
# ══════════════════════════════════════════════════════════

sample_queries = []  # auto-generated from chunks
expected_responses = []  # chunk text IS the reference


# ══════════════════════════════════════════════════════════
#  STEP 3: Run RAG pipeline for each query (build dataset)
# ══════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  RAGAS EVALUATION — EDMS RAG Pipeline")
    print("=" * 70)

    print("\nStep 1: Initializing RAG system...")
    rag = EDMSRAG()

    MAX_SAMPLES = 10  # total test cases to generate
    CHUNKS_PER_DOC = 3  # chunks to sample per document

    # ── STEP 2: Auto-generate queries from actual chunks ──
    print(f"\nStep 2: Generating test queries from document chunks...")
    conn = rag.conn

    docs = conn.execute("""
        SELECT document_key, document_name FROM rag.chunks
        WHERE chunk_type = 'child' AND embedding IS NOT NULL AND LENGTH(content) > 200
        GROUP BY document_key, document_name
    """).fetchall()

    sample_queries = []
    expected_responses = []

    for doc in docs:
        chunks = conn.execute("""
            SELECT content FROM rag.chunks
            WHERE document_key = %s AND chunk_type = 'child' AND LENGTH(content) > 200
            ORDER BY RANDOM() LIMIT %s
        """, (doc["document_key"], CHUNKS_PER_DOC)).fetchall()

        for chunk in chunks:
            chunk_text = chunk["content"][:500]

            # Groq generates ONLY the question. The chunk text IS the answer.
            q = call_groq(
                "Generate exactly 1 natural question that this text answers. Return ONLY the question.",
                f"TEXT:\n{chunk_text}",
                temperature=0.7,
            )
            if not q or len(q.strip()) < 10:
                continue

            sample_queries.append(q.strip().strip('"'))
            expected_responses.append(chunk_text)  # the chunk itself is the reference

            if len(sample_queries) >= MAX_SAMPLES:
                break
            time.sleep(0.3)
        if len(sample_queries) >= MAX_SAMPLES:
            break

    print(f"  Generated {len(sample_queries)} test queries\n")

    print(f"Step 3: Running {len(sample_queries)} queries through RAG pipeline...")
    dataset = []

    for i, (query, reference) in enumerate(zip(sample_queries, expected_responses)):
        # Retrieve
        relevant_docs = rag.get_most_relevant_docs(query, k=5)
        # Generate
        response = rag.generate_answer(query, relevant_docs)

        dataset.append({
            "user_input": query,
            "retrieved_contexts": relevant_docs,
            "response": response,
            "reference": reference,
        })

        print(f"  [{i+1}/{len(sample_queries)}] {query[:60]}...")
        time.sleep(0.5)  # rate limit

    # Save dataset for inspection
    with open("/tmp/ragas_dataset.json", "w") as f:
        json.dump(dataset, f, indent=2, default=str)
    print(f"\n  Dataset saved to /tmp/ragas_dataset.json")

    # ══════════════════════════════════════════════════════════
    #  STEP 4: Evaluate with RAGAS (using Gemini)
    # ══════════════════════════════════════════════════════════

    print(f"\nStep 3: Running RAGAS evaluation with Gemini...")

    from ragas import EvaluationDataset, evaluate
    from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness
    from ragas.llms import llm_factory
    from openai import OpenAI

    # Use OpenAI as evaluator (RAGAS native support)
    openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    evaluator_llm = llm_factory(
        "gpt-4o-mini",
        client=openai_client,
    )
    print(f"  Evaluator: OpenAI (gpt-4o-mini)")

    # Build RAGAS dataset
    evaluation_dataset = EvaluationDataset.from_list(dataset)

    # Run evaluation
    print(f"  Evaluating {len(dataset)} samples across 3 metrics...")
    print(f"  (LLMContextRecall, Faithfulness, FactualCorrectness)\n")

    result = evaluate(
        dataset=evaluation_dataset,
        metrics=[LLMContextRecall(), Faithfulness(), FactualCorrectness()],
        llm=evaluator_llm,
    )

    # ══════════════════════════════════════════════════════════
    #  STEP 5: Display results
    # ══════════════════════════════════════════════════════════

    print("\n" + "=" * 70)
    print("  RAGAS RESULTS")
    print("=" * 70 + "\n")

    print(result)

    # Detailed breakdown
    df = result.to_pandas()
    print("\nDetailed per-query scores:")
    print(df.to_string())

    # Save results
    df.to_csv("/tmp/ragas_results.csv", index=False)
    print(f"\nResults saved to /tmp/ragas_results.csv")

    # Summary with ratings
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70 + "\n")

    # Extract scores from the results CSV
    import pandas as pd
    df = pd.read_csv("/tmp/ragas_results.csv")
    metrics = {}
    for col in df.columns:
        if df[col].dtype in ('float64', 'float32'):
            valid = df[col].dropna()
            if len(valid) > 0:
                metrics[col] = valid.mean()

    for name, score in metrics.items():
        if score is None:
            score = 0
        bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
        if score >= 0.85:
            rating = "EXCELLENT"
        elif score >= 0.70:
            rating = "GOOD"
        elif score >= 0.55:
            rating = "NEEDS WORK"
        else:
            rating = "POOR"
        print(f"  {name:<25} {score:.3f}  {bar}  {rating}")

    avg = sum(v for v in metrics.values() if v) / len(metrics)
    print(f"\n  {'OVERALL':<25} {avg:.3f}")
    print()
