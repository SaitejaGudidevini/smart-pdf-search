#!/usr/bin/env python3
"""Experiment #6: Compare reranking models and strategies for RAG retrieval.

Strategies tested:
  1. No reranking (base vector retrieval only)
  2. cross-encoder/ms-marco-MiniLM-L-6-v2  (current optional reranker)
  3. BAAI/bge-reranker-base
  4. Reciprocal Rank Fusion (vector + BM25 keyword, no neural reranker)
  5. Score-weighted fusion (vector * keyword overlap)

Metrics:
  - Reranking latency (ms per query)
  - MRR, P@5, NDCG@5 improvement over base retrieval
  - Top-1 result quality shift

Usage:
  python scripts/experiment_reranking.py [--pdf path/to/file.pdf]
"""

import argparse
import math
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from experiment_harness import (
    CURATED_QUERIES,
    ExperimentResult,
    chunk_document,
    compute_chunk_stats,
    detect_paper_key,
    embed_texts,
    evaluate_retrieval,
    find_test_pdfs,
    get_queries_for_paper,
    in_memory_search,
    keyword_relevance_score,
    load_fastembed_model,
    load_structure,
    section_relevance,
    timer,
)

from chunking_pipeline import Chunk

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
INITIAL_RETRIEVAL_K = 20  # retrieve more, then rerank down to top 5
FINAL_K = 5


# ── BM25 keyword scorer (in-memory) ─────────────────────────────────

class SimpleBM25:
    """Minimal BM25 scorer for in-memory keyword matching."""

    def __init__(self, corpus: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus)
        self.avgdl = 0.0
        self.doc_freqs: dict[str, int] = Counter()
        self.doc_lens: list[int] = []
        self.tf_cache: list[dict[str, int]] = []

        for doc in corpus:
            tokens = self._tokenize(doc)
            self.doc_lens.append(len(tokens))
            tf = Counter(tokens)
            self.tf_cache.append(tf)
            for term in set(tokens):
                self.doc_freqs[term] += 1

        self.avgdl = sum(self.doc_lens) / self.corpus_size if self.corpus_size else 1.0

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"\w+", text.lower())

    def score(self, query: str, doc_idx: int) -> float:
        query_tokens = self._tokenize(query)
        doc_len = self.doc_lens[doc_idx]
        tf_doc = self.tf_cache[doc_idx]
        score = 0.0
        for term in query_tokens:
            if term not in tf_doc:
                continue
            tf = tf_doc[term]
            df = self.doc_freqs.get(term, 0)
            idf = math.log((self.corpus_size - df + 0.5) / (df + 0.5) + 1)
            tf_norm = (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl))
            score += idf * tf_norm
        return score

    def score_batch(self, query: str) -> list[float]:
        return [self.score(query, i) for i in range(self.corpus_size)]


# ── Reranking strategies ─────────────────────────────────────────────

def rerank_none(query: str, candidates: list[tuple[Chunk, float]], **kwargs) -> list[tuple[Chunk, float]]:
    """No reranking — return candidates as-is."""
    return candidates


def rerank_cross_encoder(query: str, candidates: list[tuple[Chunk, float]], model=None, **kwargs) -> list[tuple[Chunk, float]]:
    """Rerank using a cross-encoder model."""
    if model is None:
        return candidates
    pairs = [(query, chunk.text) for chunk, _ in candidates]
    scores = model.predict(pairs)
    reranked = [(chunk, float(score)) for (chunk, _), score in zip(candidates, scores)]
    reranked.sort(key=lambda x: x[1], reverse=True)
    return reranked


def rerank_rrf(query: str, candidates: list[tuple[Chunk, float]], bm25: SimpleBM25 = None, chunk_index_map: dict = None, **kwargs) -> list[tuple[Chunk, float]]:
    """Reciprocal Rank Fusion: combine vector rank with BM25 rank."""
    if bm25 is None or chunk_index_map is None:
        return candidates

    k = 60  # RRF constant
    vector_scores = {}
    for rank, (chunk, score) in enumerate(candidates):
        idx = id(chunk)
        vector_scores[idx] = 1.0 / (k + rank + 1)

    bm25_scores_raw = []
    for chunk, _ in candidates:
        doc_idx = chunk_index_map.get(id(chunk), 0)
        bm25_scores_raw.append((chunk, bm25.score(query, doc_idx)))
    bm25_scores_raw.sort(key=lambda x: x[1], reverse=True)

    bm25_rrf = {}
    for rank, (chunk, _) in enumerate(bm25_scores_raw):
        bm25_rrf[id(chunk)] = 1.0 / (k + rank + 1)

    fused = []
    for chunk, _ in candidates:
        idx = id(chunk)
        combined = vector_scores.get(idx, 0) + bm25_rrf.get(idx, 0)
        fused.append((chunk, combined))

    fused.sort(key=lambda x: x[1], reverse=True)
    return fused


def rerank_score_weighted(query: str, candidates: list[tuple[Chunk, float]], **kwargs) -> list[tuple[Chunk, float]]:
    """Weighted fusion: vector_score * (1 + keyword_overlap_bonus)."""
    reranked = []
    for chunk, vec_score in candidates:
        kw_score = keyword_relevance_score(query, chunk.text)
        combined = vec_score * (1.0 + kw_score * 0.5)
        reranked.append((chunk, combined))
    reranked.sort(key=lambda x: x[1], reverse=True)
    return reranked


# ── Evaluation ───────────────────────────────────────────────────────

def evaluate_reranker(
    name: str,
    rerank_fn,
    queries: list[tuple[str, list[str], str]],
    child_chunks: list[Chunk],
    embeddings: list[list[float]],
    model_embed,
    **rerank_kwargs,
) -> dict:
    """Evaluate one reranking strategy using section-based ground truth."""
    print(f"\n--- {name} ---")

    results_per_query = []
    query_details = []
    total_rerank_ms = 0.0

    for query, expected_sections, _paper in queries:
        # Base retrieval: get top INITIAL_RETRIEVAL_K candidates
        q_emb = embed_texts(model_embed, [query])[0]
        candidates = in_memory_search(q_emb, embeddings, child_chunks, top_k=INITIAL_RETRIEVAL_K)

        # Rerank
        with timer() as t_rerank:
            reranked = rerank_fn(query, candidates, **rerank_kwargs)
        total_rerank_ms += t_rerank["elapsed_ms"]

        top_results = reranked[:FINAL_K]
        results_per_query.append(top_results)

        relevant_count = sum(1 for chunk, _ in top_results if section_relevance(chunk, expected_sections))
        query_details.append({
            "query": query,
            "expected_sections": expected_sections,
            "rerank_time_ms": round(t_rerank["elapsed_ms"], 2),
            "top_score": round(top_results[0][1], 4) if top_results else 0,
            "top_section": top_results[0][0].metadata.get("section", "") if top_results else "",
            "top_preview": top_results[0][0].text[:100] if top_results else "",
            "relevant_in_top5": relevant_count,
        })

    retrieval = evaluate_retrieval(results_per_query, queries)
    avg_rerank_ms = total_rerank_ms / len(queries) if queries else 0

    print(f"  MRR={retrieval['mrr']}, P@5={retrieval['precision_at_5']}, NDCG@5={retrieval['ndcg_at_5']}, "
          f"avg rerank latency={avg_rerank_ms:.1f}ms")

    return {
        "strategy": name,
        "retrieval": retrieval,
        "latency": {
            "total_rerank_ms": round(total_rerank_ms, 2),
            "avg_per_query_ms": round(avg_rerank_ms, 2),
        },
        "query_details": query_details,
    }


def main():
    parser = argparse.ArgumentParser(description="Reranking strategy comparison experiment")
    parser.add_argument("--pdf", type=str, help="Path to a specific PDF to test with")
    args = parser.parse_args()

    print("=" * 60)
    print("Experiment #6: Reranking Strategy Comparison")
    print("=" * 60)

    # Find test PDF
    if args.pdf:
        pdf_path = Path(args.pdf)
    else:
        pdfs = find_test_pdfs()
        if not pdfs:
            print("ERROR: No test PDFs found. Provide one with --pdf.")
            sys.exit(1)
        pdf_path = pdfs[0]

    print(f"\nTest document: {pdf_path.name}")

    # Detect paper and select queries
    paper_key = detect_paper_key(pdf_path)
    queries = get_queries_for_paper(paper_key) if paper_key else CURATED_QUERIES
    print(f"Using {len(queries)} curated queries" +
          (f" for '{paper_key}'" if paper_key else " (all papers)"))

    # Extract, chunk, embed
    with timer("structure") as t_struct:
        structure = load_structure(pdf_path)
    print(f"Structure: {len(structure.pages)} pages")

    with timer("chunking") as t_chunk:
        chunks = chunk_document(structure, pdf_path.name)
    chunk_stats = compute_chunk_stats(chunks)

    child_chunks = [c for c in chunks if c.metadata.get("chunk_type") == "child"]
    child_texts = [c.text for c in child_chunks]
    print(f"Chunks: {chunk_stats['child_chunks']} children for retrieval")

    print(f"\nLoading embedding model: {EMBEDDING_MODEL}")
    with timer("model_load") as t_model:
        model_embed = load_fastembed_model(EMBEDDING_MODEL)

    with timer("embedding") as t_emb:
        embeddings = embed_texts(model_embed, child_texts)
    print(f"Embedded {len(embeddings)} chunks")

    # Build BM25 index for RRF strategy
    print("Building BM25 index...")
    bm25 = SimpleBM25(child_texts)
    chunk_index_map = {id(chunk): i for i, chunk in enumerate(child_chunks)}

    # Load cross-encoder models
    cross_encoders = {}
    ce_models = [
        ("ms-marco-MiniLM-L-6-v2", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
        ("bge-reranker-base", "BAAI/bge-reranker-base"),
    ]
    for name, model_id in ce_models:
        try:
            from sentence_transformers import CrossEncoder
            print(f"Loading cross-encoder: {model_id}...")
            cross_encoders[name] = CrossEncoder(model_id)
        except ImportError:
            print(f"  sentence-transformers not installed, skipping {name}")
        except Exception as e:
            print(f"  Failed to load {model_id}: {e}")

    # Define strategies
    strategies = [
        ("no_reranking", rerank_none, {}),
        ("rrf_vector_bm25", rerank_rrf, {"bm25": bm25, "chunk_index_map": chunk_index_map}),
        ("score_weighted_fusion", rerank_score_weighted, {}),
    ]

    for name, model in cross_encoders.items():
        strategies.append((
            f"cross_encoder_{name}",
            rerank_cross_encoder,
            {"model": model},
        ))

    # Evaluate each strategy
    results = []
    for name, rerank_fn, kwargs in strategies:
        result = evaluate_reranker(
            name, rerank_fn, queries,
            child_chunks, embeddings, model_embed,
            **kwargs,
        )
        results.append(result)

    # Compute improvement over baseline
    baseline_mrr = results[0]["retrieval"]["mrr"] if results else 0
    baseline_ndcg = results[0]["retrieval"]["ndcg_at_5"] if results else 0

    for r in results:
        mrr_delta = r["retrieval"]["mrr"] - baseline_mrr
        ndcg_delta = r["retrieval"]["ndcg_at_5"] - baseline_ndcg
        r["improvement"] = {
            "mrr_delta": round(mrr_delta, 4),
            "ndcg_delta": round(ndcg_delta, 4),
            "mrr_pct_change": round(mrr_delta / baseline_mrr * 100, 1) if baseline_mrr else 0,
            "ndcg_pct_change": round(ndcg_delta / baseline_ndcg * 100, 1) if baseline_ndcg else 0,
        }

    # Summary
    valid = [r for r in results if r["retrieval"]["mrr"] > 0]
    summary = {}
    if valid:
        best_quality = max(valid, key=lambda r: r["retrieval"]["mrr"])
        fastest = min(valid, key=lambda r: r["latency"]["avg_per_query_ms"])
        best_tradeoff = max(valid, key=lambda r: (
            r["retrieval"]["mrr"] * 0.7
            - (r["latency"]["avg_per_query_ms"] / 1000) * 0.3
        ))
        summary = {
            "best_quality": best_quality["strategy"],
            "fastest": fastest["strategy"],
            "best_quality_latency_tradeoff": best_tradeoff["strategy"],
        }

    # Save
    experiment = ExperimentResult(
        experiment_name="reranking_strategy_comparison",
        config={
            "test_document": pdf_path.name,
            "paper_key": paper_key,
            "embedding_model": EMBEDDING_MODEL,
            "initial_retrieval_k": INITIAL_RETRIEVAL_K,
            "final_k": FINAL_K,
            "strategies_tested": [s[0] for s, _, _ in strategies],
            "cross_encoders_loaded": list(cross_encoders.keys()),
            "num_queries": len(queries),
            "ground_truth": "section-based (curated expected sections per query)",
        },
        metrics={
            "chunk_stats": chunk_stats,
            "baseline_mrr": baseline_mrr,
            "baseline_ndcg": baseline_ndcg,
            "summary": summary,
        },
        timings={
            "structure_ms": t_struct["elapsed_ms"],
            "chunking_ms": t_chunk["elapsed_ms"],
            "model_load_ms": t_model["elapsed_ms"],
            "embedding_ms": t_emb["elapsed_ms"],
        },
        details=results,
    )

    output = experiment.save()

    # Print comparison
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    print(f"  {'Strategy':30s} {'MRR':>6s} {'P@5':>6s} {'NDCG':>6s} {'MRR +/-':>8s} {'Lat(ms)':>8s}")
    print(f"  {'-'*30} {'-'*6} {'-'*6} {'-'*6} {'-'*8} {'-'*8}")
    for r in results:
        ret = r["retrieval"]
        imp = r["improvement"]
        lat = r["latency"]["avg_per_query_ms"]
        delta = f"{imp['mrr_delta']:+.3f}" if imp["mrr_delta"] != 0 else "    base"
        print(f"  {r['strategy']:30s} {ret['mrr']:6.3f} {ret['precision_at_5']:6.3f} "
              f"{ret['ndcg_at_5']:6.3f} {delta:>8s} {lat:8.1f}")

    if summary:
        print(f"\n  Best tradeoff: {summary['best_quality_latency_tradeoff']}")

    return experiment.to_dict()


if __name__ == "__main__":
    main()
