#!/usr/bin/env python3
"""Experiment #3: Benchmark the current RAG pipeline's end-to-end performance.

Measures the production pipeline as-is:
  - Structure extraction (pymupdf4llm)
  - Chunking (semantic + section-aware + parent-child)
  - Embedding (fastembed all-MiniLM-L6-v2, 384d)
  - Hybrid search (vector + keyword + RRF fusion)
  - Optional reranking (cross-encoder if available)

Uses curated queries with section-based ground truth from the research team.
Compares against baseline chunk data from different extractors.

Usage:
  python scripts/benchmark_current.py [--pdf path/to/file.pdf] [--all]
"""

import argparse
import sys
from pathlib import Path

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
    in_memory_search,
    load_baseline_chunks,
    load_fastembed_model,
    load_structure,
    section_relevance,
    timer,
)

from chunking_pipeline import Chunk

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def benchmark_one_pdf(pdf_path: Path, model) -> dict:
    """Run full pipeline benchmark on a single PDF."""
    print(f"\n{'─' * 50}")
    print(f"Benchmarking: {pdf_path.name}")
    print(f"{'─' * 50}")

    # Detect paper and select queries
    paper_key = detect_paper_key(pdf_path)
    queries = get_queries_for_paper(paper_key) if paper_key else CURATED_QUERIES
    print(f"  Paper: {paper_key or 'unknown'}, {len(queries)} curated queries")

    # 1. Structure extraction
    with timer("structure_extraction") as t_struct:
        structure = load_structure(pdf_path)
    print(f"  Pages: {len(structure.pages)}, Type: {structure.doc_type}, Title: {structure.title[:60]}")

    # 2. Chunking
    with timer("chunking") as t_chunk:
        chunks = chunk_document(structure, pdf_path.name)
    chunk_stats = compute_chunk_stats(chunks)
    print(f"  Chunks: {chunk_stats['total_chunks']} "
          f"({chunk_stats['parent_chunks']} parents, {chunk_stats['child_chunks']} children)")
    print(f"  Avg length: {chunk_stats['avg_length_chars']} chars, "
          f"Mid-sentence breaks: {chunk_stats['mid_sentence_break_pct']}%")

    # 3. Compare with baseline chunk data (if available)
    baseline_comparison = {}
    if paper_key == "deepseekmath":
        for variant in ["deepseekmath_default", "deepseekmath_pymupdf4llm",
                        "deepseekmath_docling", "deepseekmath_unstructured"]:
            baseline = load_baseline_chunks(variant)
            if baseline:
                baseline_comparison[variant] = {
                    "chunk_count": len(baseline),
                    "child_chunks": sum(1 for c in baseline if c.get("chunk_type") == "child"),
                    "avg_char_count": round(
                        sum(c.get("char_count", len(c.get("text", ""))) for c in baseline) / len(baseline), 1
                    ) if baseline else 0,
                }
        if baseline_comparison:
            print(f"  Baseline comparisons loaded: {list(baseline_comparison.keys())}")

    # 4. Embedding
    child_chunks = [c for c in chunks if c.metadata.get("chunk_type") == "child"]
    child_texts = [c.text for c in child_chunks]

    if not child_texts:
        print("  WARNING: No child chunks to embed.")
        return {"document": pdf_path.name, "error": "no_child_chunks"}

    with timer("embedding") as t_emb:
        embeddings = embed_texts(model, child_texts)
    emb_stats = compute_embedding_stats(embeddings, t_emb["elapsed_ms"])
    print(f"  Embedded {emb_stats['count']} chunks: {emb_stats['per_embedding_ms']:.1f}ms/chunk, "
          f"{emb_stats['embeddings_per_sec']:.0f}/sec")

    # 5. Retrieval evaluation with section-based ground truth
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
            "search_time_ms": round(t_q["elapsed_ms"], 2),
            "top_similarity": round(results[0][1], 4) if results else 0,
            "top_section": results[0][0].metadata.get("section", "") if results else "",
            "top_preview": results[0][0].text[:120] if results else "",
            "relevant_in_top5": relevant_count,
        })

    retrieval = evaluate_retrieval(results_per_query, queries)
    print(f"  Retrieval: MRR={retrieval['mrr']}, P@5={retrieval['precision_at_5']}, NDCG@5={retrieval['ndcg_at_5']}")

    # 6. Section coverage: how many unique sections are represented in results?
    sections_hit = set()
    for results in results_per_query:
        for chunk, _ in results:
            sec = chunk.metadata.get("section", "")
            if sec:
                sections_hit.add(sec)

    total_sections = len({c.metadata.get("section", "") for c in child_chunks if c.metadata.get("section")})
    coverage = len(sections_hit) / total_sections if total_sections else 0

    # 7. Embedding diversity: average pairwise similarity in result sets
    avg_result_diversity = 0.0
    diversity_samples = 0
    for results in results_per_query[:5]:
        result_indices = []
        for chunk, _ in results:
            try:
                idx = child_chunks.index(chunk)
                result_indices.append(idx)
            except ValueError:
                pass
        for i in range(len(result_indices)):
            for j in range(i + 1, len(result_indices)):
                avg_result_diversity += cosine_similarity(
                    embeddings[result_indices[i]], embeddings[result_indices[j]]
                )
                diversity_samples += 1
    avg_result_diversity = avg_result_diversity / diversity_samples if diversity_samples else 0

    return {
        "document": pdf_path.name,
        "paper_key": paper_key,
        "doc_type": structure.doc_type,
        "pages": len(structure.pages),
        "title": structure.title,
        "chunk_stats": chunk_stats,
        "baseline_comparison": baseline_comparison,
        "embedding_stats": emb_stats,
        "retrieval": {
            **retrieval,
            "section_coverage": round(coverage, 3),
            "sections_hit": len(sections_hit),
            "total_sections": total_sections,
            "avg_result_diversity": round(avg_result_diversity, 4),
        },
        "timings": {
            "structure_extraction_ms": t_struct["elapsed_ms"],
            "chunking_ms": t_chunk["elapsed_ms"],
            "embedding_ms": t_emb["elapsed_ms"],
            "total_pipeline_ms": round(
                t_struct["elapsed_ms"] + t_chunk["elapsed_ms"] + t_emb["elapsed_ms"], 2
            ),
        },
        "query_details": query_details,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark current RAG pipeline")
    parser.add_argument("--pdf", type=str, help="Path to a specific PDF to benchmark")
    parser.add_argument("--all", action="store_true", help="Benchmark all available PDFs")
    args = parser.parse_args()

    print("=" * 60)
    print("Experiment #3: Current Pipeline Benchmark")
    print("=" * 60)

    # Collect PDFs
    if args.pdf:
        pdf_paths = [Path(args.pdf)]
    elif args.all:
        pdf_paths = find_test_pdfs()
    else:
        pdf_paths = find_test_pdfs()[:1]

    if not pdf_paths:
        print("ERROR: No test PDFs found. Provide one with --pdf.")
        sys.exit(1)

    print(f"PDFs to benchmark: {len(pdf_paths)}")
    for p in pdf_paths:
        paper_key = detect_paper_key(p)
        label = f" [{paper_key}]" if paper_key else ""
        print(f"  - {p.name} ({p.stat().st_size // 1024}KB){label}")

    # Load embedding model once
    print(f"\nLoading embedding model: {EMBEDDING_MODEL}")
    with timer("model_load") as t_model:
        model = load_fastembed_model(EMBEDDING_MODEL)
    print(f"  Model loaded in {t_model['elapsed_ms']:.0f}ms")

    # Benchmark each PDF
    doc_results = []
    for pdf_path in pdf_paths:
        result = benchmark_one_pdf(pdf_path, model)
        doc_results.append(result)

    # Aggregate metrics across documents
    valid = [r for r in doc_results if "error" not in r]
    aggregate = {}
    if valid:
        aggregate = {
            "documents_benchmarked": len(valid),
            "avg_mrr": round(sum(r["retrieval"]["mrr"] for r in valid) / len(valid), 4),
            "avg_p5": round(sum(r["retrieval"]["precision_at_5"] for r in valid) / len(valid), 4),
            "avg_ndcg": round(sum(r["retrieval"]["ndcg_at_5"] for r in valid) / len(valid), 4),
            "avg_embedding_speed": round(
                sum(r["embedding_stats"]["embeddings_per_sec"] for r in valid) / len(valid), 1
            ),
            "avg_pipeline_time_ms": round(
                sum(r["timings"]["total_pipeline_ms"] for r in valid) / len(valid), 1
            ),
            "total_chunks": sum(r["chunk_stats"]["total_chunks"] for r in valid),
            "avg_mid_sentence_breaks_pct": round(
                sum(r["chunk_stats"]["mid_sentence_break_pct"] for r in valid) / len(valid), 1
            ),
            "ground_truth": "section-based (curated expected sections per query)",
        }

    # Save
    experiment = ExperimentResult(
        experiment_name="current_pipeline_benchmark",
        config={
            "embedding_model": EMBEDDING_MODEL,
            "embedding_dimensions": 384,
            "chunking": "semantic + section-aware + parent-child",
            "search": "hybrid (vector + keyword + RRF)",
            "documents": [p.name for p in pdf_paths],
            "ground_truth": "section-based (curated expected sections per query)",
        },
        metrics=aggregate,
        timings={"model_load_ms": t_model["elapsed_ms"]},
        details=doc_results,
    )

    output = experiment.save()

    # Print summary
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)

    if aggregate:
        print(f"  Documents: {aggregate['documents_benchmarked']}")
        print(f"  Total chunks: {aggregate['total_chunks']}")
        print(f"  Avg MRR: {aggregate['avg_mrr']}")
        print(f"  Avg P@5: {aggregate['avg_p5']}")
        print(f"  Avg NDCG@5: {aggregate['avg_ndcg']}")
        print(f"  Embedding speed: {aggregate['avg_embedding_speed']:.0f} embeddings/sec")
        print(f"  Avg pipeline time: {aggregate['avg_pipeline_time_ms']:.0f}ms")
        print(f"  Mid-sentence breaks: {aggregate['avg_mid_sentence_breaks_pct']}%")

    for r in valid:
        print(f"\n  {r['document']}:")
        print(f"    Paper={r.get('paper_key', 'unknown')}, Type={r['doc_type']}, "
              f"Pages={r['pages']}, Chunks={r['chunk_stats']['total_chunks']}")
        print(f"    MRR={r['retrieval']['mrr']}, P@5={r['retrieval']['precision_at_5']}, "
              f"Section coverage={r['retrieval']['section_coverage']}")

        if r.get("baseline_comparison"):
            print(f"    Baseline chunk counts:")
            for variant, stats in r["baseline_comparison"].items():
                print(f"      {variant}: {stats['chunk_count']} chunks "
                      f"({stats['child_chunks']} children, avg {stats['avg_char_count']} chars)")

    return experiment.to_dict()


if __name__ == "__main__":
    main()
