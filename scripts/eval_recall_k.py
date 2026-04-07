"""Recall@K Evaluation — auto-generates test queries from chunks, runs retrieval, measures quality.

Usage:
  docker exec rag-sidecar python /app/eval_recall_k.py

Or locally:
  RAG_PGHOST=localhost python scripts/eval_recall_k.py
"""

import json
import os
import sys
import time
from collections import defaultdict

import httpx
import psycopg
from psycopg.rows import dict_row

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "smart-search"))

# Config
PG_HOST = os.environ.get("RAG_PGHOST", "localhost")
PG_PORT = int(os.environ.get("RAG_PGPORT", "5432"))
PG_DB = os.environ.get("RAG_PGDATABASE", "mayan")
PG_USER = os.environ.get("RAG_PGUSER", "mayan")
PG_PASS = os.environ.get("RAG_PGPASSWORD", "mayan")
GROQ_KEY = os.environ.get("GROQ_API_KEY", "")

QUERIES_PER_CHUNK = 2
CHUNKS_PER_DOC = 10  # sample this many chunks per document
K_VALUES = [1, 3, 5, 10]


def get_connection():
    return psycopg.connect(
        host=PG_HOST, port=PG_PORT, dbname=PG_DB,
        user=PG_USER, password=PG_PASS, row_factory=dict_row,
    )


def call_groq(system: str, user: str) -> str | None:
    if not GROQ_KEY:
        return None
    try:
        resp = httpx.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_KEY}", "Content-Type": "application/json"},
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "max_tokens": 500, "temperature": 0.7,
            },
            timeout=30,
        )
        if resp.status_code == 200:
            return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"  Groq error: {e}")
    return None


# ══════════════════════════════════════════════════════════
#  STEP 1: Generate test queries from chunks
# ══════════════════════════════════════════════════════════

def generate_test_queries(conn) -> list[dict]:
    """Sample chunks from each document and generate questions via LLM."""

    # Get all documents
    docs = conn.execute("""
        SELECT document_key, document_name, COUNT(*) as chunk_count
        FROM rag.chunks WHERE chunk_type = 'child' AND embedding IS NOT NULL
        GROUP BY document_key, document_name
    """).fetchall()

    print(f"\n{'='*80}")
    print(f"  STEP 1: Auto-generate test queries from {len(docs)} documents")
    print(f"{'='*80}\n")

    test_cases = []

    for doc in docs:
        # Sample random chunks from this document
        chunks = conn.execute("""
            SELECT id::text as chunk_id, content, section, page_number
            FROM rag.chunks
            WHERE document_key = %s AND chunk_type = 'child'
              AND embedding IS NOT NULL AND LENGTH(content) > 100
            ORDER BY RANDOM()
            LIMIT %s
        """, (doc["document_key"], CHUNKS_PER_DOC)).fetchall()

        print(f"  {doc['document_name']} ({doc['chunk_count']} chunks, sampling {len(chunks)})")

        for chunk in chunks:
            content = chunk["content"][:600]

            prompt = (
                f"Given this text passage from a document, generate exactly {QUERIES_PER_CHUNK} questions "
                f"that this passage would be the ideal answer for.\n\n"
                f"Rules:\n"
                f"- Questions should be natural (what a real user would ask)\n"
                f"- Do NOT copy exact phrases from the passage\n"
                f"- Each question should have a clear answer from the passage\n"
                f"- Return ONLY a JSON array of strings, no explanation\n\n"
                f"PASSAGE:\n{content}\n\n"
                f"JSON array of {QUERIES_PER_CHUNK} questions:"
            )

            raw = call_groq("You generate evaluation questions for a search system. Return only JSON.", prompt)
            if not raw:
                continue

            # Parse JSON
            try:
                raw = raw.strip()
                if raw.startswith("```"):
                    raw = "\n".join(raw.split("\n")[1:])
                    if raw.endswith("```"):
                        raw = raw[:-3]
                questions = json.loads(raw)
                if not isinstance(questions, list):
                    continue
            except json.JSONDecodeError:
                # Try to find JSON array in text
                import re
                match = re.search(r"\[.*\]", raw, re.DOTALL)
                if match:
                    try:
                        questions = json.loads(match.group())
                    except json.JSONDecodeError:
                        continue
                else:
                    continue

            for q in questions[:QUERIES_PER_CHUNK]:
                if isinstance(q, str) and len(q) > 10:
                    test_cases.append({
                        "query": q,
                        "expected_doc_key": doc["document_key"],
                        "expected_doc_name": doc["document_name"],
                        "source_chunk_id": chunk["chunk_id"],
                        "source_section": chunk["section"],
                        "source_page": chunk["page_number"],
                    })

            # Rate limit
            time.sleep(0.5)

    print(f"\n  Generated {len(test_cases)} test queries total\n")
    return test_cases


# ══════════════════════════════════════════════════════════
#  STEP 2: Run retrieval and compute Recall@K
# ══════════════════════════════════════════════════════════

def evaluate_recall(conn, test_cases: list[dict], model) -> dict:
    """Run retrieval for each test query and compute Recall@K."""

    print(f"{'='*80}")
    print(f"  STEP 2: Running retrieval on {len(test_cases)} queries")
    print(f"{'='*80}\n")

    max_k = max(K_VALUES)

    # Track results
    doc_hits = {k: [] for k in K_VALUES}  # document-level recall
    chunk_hits = {k: [] for k in K_VALUES}  # chunk-level recall (exact chunk found)
    per_doc_stats = defaultdict(lambda: {"total": 0, "doc_hit_1": 0, "chunk_hit_1": 0})

    for i, tc in enumerate(test_cases):
        query = tc["query"]
        expected_key = tc["expected_doc_key"]
        expected_chunk_id = tc["source_chunk_id"]

        # Embed query
        query_emb = list(model.embed([query]))[0].tolist()

        # Retrieve top K (unscoped — all documents)
        results = conn.execute("""
            SELECT id::text as chunk_id, document_key, document_name,
                   1-(embedding <=> %s::vector) as score
            FROM rag.chunks
            WHERE chunk_type = 'child' AND embedding IS NOT NULL
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """, (query_emb, query_emb, max_k)).fetchall()

        for k in K_VALUES:
            top_k = results[:k]
            # Document hit: did the right document appear?
            doc_found = any(r["document_key"] == expected_key for r in top_k)
            doc_hits[k].append(1.0 if doc_found else 0.0)

            # Chunk hit: did the exact source chunk appear?
            chunk_found = any(r["chunk_id"] == expected_chunk_id for r in top_k)
            chunk_hits[k].append(1.0 if chunk_found else 0.0)

        # Per-document tracking
        doc_name = tc["expected_doc_name"]
        per_doc_stats[doc_name]["total"] += 1
        if any(r["document_key"] == expected_key for r in results[:1]):
            per_doc_stats[doc_name]["doc_hit_1"] += 1
        if any(r["chunk_id"] == expected_chunk_id for r in results[:1]):
            per_doc_stats[doc_name]["chunk_hit_1"] += 1

        # Progress
        if (i + 1) % 10 == 0 or i == len(test_cases) - 1:
            print(f"  Processed {i+1}/{len(test_cases)} queries")

    return {
        "doc_hits": doc_hits,
        "chunk_hits": chunk_hits,
        "per_doc_stats": per_doc_stats,
    }


# ══════════════════════════════════════════════════════════
#  STEP 3: Print results
# ══════════════════════════════════════════════════════════

def print_results(results: dict, test_cases: list[dict]):
    doc_hits = results["doc_hits"]
    chunk_hits = results["chunk_hits"]
    per_doc_stats = results["per_doc_stats"]

    print(f"\n{'='*80}")
    print(f"  RECALL@K RESULTS ({len(test_cases)} queries)")
    print(f"{'='*80}\n")

    # Overall scores
    print(f"  {'Metric':<30}", end="")
    for k in K_VALUES:
        print(f"  K={k:<4}", end="")
    print()
    print(f"  {'-'*60}")

    print(f"  {'Document Recall@K':<30}", end="")
    for k in K_VALUES:
        avg = sum(doc_hits[k]) / len(doc_hits[k]) if doc_hits[k] else 0
        print(f"  {avg:.3f} ", end="")
    print()

    print(f"  {'Exact Chunk Recall@K':<30}", end="")
    for k in K_VALUES:
        avg = sum(chunk_hits[k]) / len(chunk_hits[k]) if chunk_hits[k] else 0
        print(f"  {avg:.3f} ", end="")
    print()

    # Rating
    print(f"\n  Rating:")
    for k in K_VALUES:
        doc_avg = sum(doc_hits[k]) / len(doc_hits[k]) if doc_hits[k] else 0
        if doc_avg >= 0.90:
            rating = "EXCELLENT — ship it"
        elif doc_avg >= 0.80:
            rating = "GOOD — production ready"
        elif doc_avg >= 0.70:
            rating = "OK — acceptable for internal tools"
        elif doc_avg >= 0.60:
            rating = "NEEDS WORK"
        else:
            rating = "POOR — do not ship"
        print(f"    Doc Recall@{k} = {doc_avg:.3f} → {rating}")

    # Per-document breakdown
    print(f"\n  Per-document breakdown (Recall@1):")
    print(f"  {'Document':<55} {'Queries':<10} {'Doc@1':<10} {'Chunk@1':<10}")
    print(f"  {'-'*85}")
    for doc_name, stats in sorted(per_doc_stats.items()):
        total = stats["total"]
        doc_r1 = stats["doc_hit_1"] / total if total else 0
        chunk_r1 = stats["chunk_hit_1"] / total if total else 0
        print(f"  {doc_name:<55} {total:<10} {doc_r1:<10.3f} {chunk_r1:<10.3f}")

    # Show some failures
    print(f"\n  Sample failures (Doc Recall@1 misses):")
    shown = 0
    for i, tc in enumerate(test_cases):
        if doc_hits[1][i] == 0.0 and shown < 5:
            print(f"    Q: \"{tc['query'][:70]}\"")
            print(f"       Expected: {tc['expected_doc_name']}")
            shown += 1
    if shown == 0:
        print(f"    None — all queries found the right document at #1!")


# ══════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("  RAG RECALL@K EVALUATION")
    print("  Auto-generates queries from chunks, runs retrieval, measures quality")
    print("=" * 80)

    conn = get_connection()

    # Load embedding model
    from fastembed import TextEmbedding
    print("\nLoading embedding model...")
    model = TextEmbedding("BAAI/bge-base-en-v1.5")
    print("Model loaded.\n")

    # Step 1: Generate test queries
    test_cases = generate_test_queries(conn)

    if not test_cases:
        print("No test queries generated. Check GROQ_API_KEY.")
        sys.exit(1)

    # Save test cases for reproducibility
    with open("/tmp/recall_test_cases.json", "w") as f:
        json.dump(test_cases, f, indent=2)
    print(f"  Test cases saved to /tmp/recall_test_cases.json")

    # Step 2: Run evaluation
    results = evaluate_recall(conn, test_cases, model)

    # Step 3: Print results
    print_results(results, test_cases)

    conn.close()
