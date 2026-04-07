#!/usr/bin/env python3
"""Experiment #4: Compare alternative embedding models for RAG retrieval.

Models tested:
  1. sentence-transformers/all-MiniLM-L6-v2   (384d, current default)
  2. BAAI/bge-small-en-v1.5                    (384d)
  3. nomic-ai/nomic-embed-text-v1.5            (768d)
  4. jinaai/jina-embeddings-v2-small-en        (512d)

Metrics:
  - Embedding speed (embeddings/sec)
  - Vector dimensions
  - Retrieval quality: MRR, P@5, NDCG@5 with section-based ground truth
  - Intra-cluster similarity (do chunks from the same section cluster together?)

Usage:
  python scripts/experiment_embeddings.py [--pdf path/to/file.pdf]
"""

import argparse
import sys
from pathlib import Path

# Bootstrap imports
sys.path.insert(0, str(Path(__file__).resolve().parent))
from experiment_harness import (
    CURATED_QUERIES,
    ExperimentResult,
    chunk_document,
    compute_chunk_stats,
    compute_embedding_stats,
    cosine_similarity,
    detect_paper_key,
    embed_texts,
    evaluate_retrieval,
    find_test_pdfs,
    get_queries_for_paper,
    graded_relevance,
    in_memory_search,
    load_fastembed_model,
    load_structure,
    section_relevance,
    timer,
)

# Models to benchmark — fastembed-compatible names
MODELS = [
    {
        "name": "all-MiniLM-L6-v2",
        "fastembed_id": "sentence-transformers/all-MiniLM-L6-v2",
        "dims": 384,
        "note": "Current default in pipeline",
    },
    {
        "name": "bge-small-en-v1.5",
        "fastembed_id": "BAAI/bge-small-en-v1.5",
        "dims": 384,
        "note": "Strong retrieval model, same dimensions",
    },
    {
        "name": "nomic-embed-text-v1.5",
        "fastembed_id": "nomic-ai/nomic-embed-text-v1.5",
        "dims": 768,
        "note": "Larger model, higher dimensions",
    },
    {
        "name": "jina-embeddings-v2-small-en",
        "fastembed_id": "jinaai/jina-embeddings-v2-small-en",
        "dims": 512,
        "note": "Jina small model, 512 dimensions",
    },
]


def evaluate_model(
    model_config: dict,
    chunks,
    child_texts: list[str],
    queries: list[tuple[str, list[str], str]],
) -> dict:
    """Evaluate a single embedding model on speed and retrieval quality."""
    model_name = model_config["fastembed_id"]
    print(f"\n--- {model_config['name']} ---")

    # Load model
    with timer("model_load") as t_load:
        try:
            model = load_fastembed_model(model_name)
        except Exception as e:
            print(f"  SKIP: Failed to load {model_name}: {e}")
            return {"name": model_config["name"], "error": str(e)}

    # Embed all child chunks
    with timer("embed_chunks") as t_embed:
        embeddings = embed_texts(model, child_texts)

    emb_stats = compute_embedding_stats(embeddings, t_embed["elapsed_ms"])
    print(f"  Embedded {emb_stats['count']} chunks in {emb_stats['total_time_ms']:.0f}ms "
          f"({emb_stats['embeddings_per_sec']:.0f}/sec)")

    # Retrieval evaluation using curated queries with section ground truth
    child_chunks = [c for c in chunks if c.metadata.get("chunk_type") == "child"]
    results_per_query = []
    query_details = []

    for query, expected_sections, _paper in queries:
        with timer() as t_q:
            q_emb = embed_texts(model, [query])[0]
            results = in_memory_search(q_emb, embeddings, child_chunks, top_k=5)

        results_per_query.append(results)

        relevant_count = sum(1 for chunk, _ in results if section_relevance(chunk, expected_sections))
        query_details.append({
            "query": query,
            "expected_sections": expected_sections,
            "search_time_ms": t_q["elapsed_ms"],
            "top_result_score": round(results[0][1], 4) if results else 0,
            "top_result_section": results[0][0].metadata.get("section", "") if results else "",
            "top_result_preview": results[0][0].text[:120] if results else "",
            "relevant_in_top5": relevant_count,
        })

    retrieval = evaluate_retrieval(results_per_query, queries)
    print(f"  MRR={retrieval['mrr']}, P@5={retrieval['precision_at_5']}, NDCG@5={retrieval['ndcg_at_5']}")

    # Intra-section coherence: chunks from same section should be more similar
    section_sims = []
    cross_sims = []
    for i in range(min(len(embeddings), 50)):
        for j in range(i + 1, min(len(embeddings), 50)):
            sim = cosine_similarity(embeddings[i], embeddings[j])
            sec_i = child_chunks[i].metadata.get("section", "")
            sec_j = child_chunks[j].metadata.get("section", "")
            if sec_i and sec_j and sec_i == sec_j:
                section_sims.append(sim)
            else:
                cross_sims.append(sim)

    coherence = {
        "intra_section_sim": round(sum(section_sims) / len(section_sims), 4) if section_sims else 0,
        "cross_section_sim": round(sum(cross_sims) / len(cross_sims), 4) if cross_sims else 0,
    }
    if section_sims and cross_sims:
        coherence["separation_ratio"] = round(coherence["intra_section_sim"] / coherence["cross_section_sim"], 3)

    print(f"  Section coherence: intra={coherence['intra_section_sim']}, "
          f"cross={coherence['cross_section_sim']}")

    return {
        "name": model_config["name"],
        "fastembed_id": model_config["fastembed_id"],
        "note": model_config["note"],
        "model_load_ms": t_load["elapsed_ms"],
        "embedding_stats": emb_stats,
        "retrieval": retrieval,
        "coherence": coherence,
        "query_details": query_details,
    }


def main():
    parser = argparse.ArgumentParser(description="Embedding model comparison experiment")
    parser.add_argument("--pdf", type=str, help="Path to a specific PDF to test with")
    args = parser.parse_args()

    print("=" * 60)
    print("Experiment #4: Embedding Model Comparison")
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

    # Detect paper and select appropriate queries
    paper_key = detect_paper_key(pdf_path)
    queries = get_queries_for_paper(paper_key) if paper_key else CURATED_QUERIES
    print(f"Using {len(queries)} curated queries" +
          (f" for '{paper_key}'" if paper_key else " (all papers)"))

    # Extract and chunk
    with timer("structure_extraction") as t_struct:
        structure = load_structure(pdf_path)
    print(f"Extracted structure: {len(structure.pages)} pages, type={structure.doc_type}")

    with timer("chunking") as t_chunk:
        chunks = chunk_document(structure, pdf_path.name)
    chunk_stats = compute_chunk_stats(chunks)
    print(f"Chunks: {chunk_stats['total_chunks']} ({chunk_stats['child_chunks']} children)")

    # Get child chunk texts for embedding
    child_chunks = [c for c in chunks if c.metadata.get("chunk_type") == "child"]
    child_texts = [c.text for c in child_chunks]

    if not child_texts:
        print("ERROR: No child chunks to embed.")
        sys.exit(1)

    # Evaluate each model
    model_results = []
    for model_config in MODELS:
        result = evaluate_model(model_config, chunks, child_texts, queries)
        model_results.append(result)

    # Build comparison summary
    valid_results = [r for r in model_results if "error" not in r]
    if valid_results:
        best_mrr = max(valid_results, key=lambda r: r["retrieval"]["mrr"])
        best_speed = max(valid_results, key=lambda r: r["embedding_stats"]["embeddings_per_sec"])
        best_coherence = max(
            valid_results,
            key=lambda r: r["coherence"].get("separation_ratio", 0),
        )

        summary = {
            "best_retrieval_quality": best_mrr["name"],
            "best_speed": best_speed["name"],
            "best_section_coherence": best_coherence["name"],
            "recommendation": _pick_recommendation(valid_results),
        }
    else:
        summary = {"error": "No models could be evaluated"}

    # Save results
    experiment = ExperimentResult(
        experiment_name="embedding_model_comparison",
        config={
            "test_document": pdf_path.name,
            "paper_key": paper_key,
            "models_tested": [m["name"] for m in MODELS],
            "num_queries": len(queries),
            "ground_truth": "section-based (curated expected sections per query)",
        },
        metrics={
            "chunk_stats": chunk_stats,
            "summary": summary,
        },
        timings={
            "structure_extraction_ms": t_struct["elapsed_ms"],
            "chunking_ms": t_chunk["elapsed_ms"],
        },
        details=model_results,
    )

    output = experiment.save()

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for r in valid_results:
        print(f"  {r['name']:35s} MRR={r['retrieval']['mrr']:.3f}  "
              f"P@5={r['retrieval']['precision_at_5']:.3f}  "
              f"Speed={r['embedding_stats']['embeddings_per_sec']:.0f}/s  "
              f"Dims={r['embedding_stats']['dimensions']}")

    if "recommendation" in summary:
        print(f"\n  Recommendation: {summary['recommendation']}")

    return experiment.to_dict()


def _pick_recommendation(results: list[dict]) -> str:
    """Score models on balanced criteria and recommend best overall."""
    scored = []
    for r in results:
        quality = r["retrieval"]["mrr"] * 0.5 + r["retrieval"]["ndcg_at_5"] * 0.3
        coherence = r["coherence"].get("separation_ratio", 1.0) * 0.1
        max_speed = max(x["embedding_stats"]["embeddings_per_sec"] for x in results)
        speed_norm = (r["embedding_stats"]["embeddings_per_sec"] / max_speed) * 0.1
        composite = quality + coherence + speed_norm
        scored.append((r["name"], composite))

    scored.sort(key=lambda x: x[1], reverse=True)
    return f"{scored[0][0]} (composite score: {scored[0][1]:.3f})"


if __name__ == "__main__":
    main()
