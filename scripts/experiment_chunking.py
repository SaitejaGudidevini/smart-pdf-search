#!/usr/bin/env python3
"""Experiment #5: Compare chunking strategies for RAG retrieval.

Strategies tested:
  1. Current pipeline — semantic splitter + section-aware + parent-child
  2. Fixed-size 256 chars with 10% overlap
  3. Fixed-size 512 chars with 15% overlap
  4. Fixed-size 1024 chars with 20% overlap
  5. Recursive character splitting (langchain)
  6. Section-aware (one chunk per section, no sub-splitting)

Metrics:
  - Chunk count, avg/min/max size
  - Mid-sentence break percentage (boundary quality)
  - Retrieval quality: MRR, P@5, NDCG@5 with section-based ground truth
  - Embedding speed (constant model, varying chunk sizes)

Usage:
  python scripts/experiment_chunking.py [--pdf path/to/file.pdf]
"""

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from experiment_harness import (
    CURATED_QUERIES,
    ExperimentResult,
    chunk_document,
    compute_chunk_stats,
    compute_embedding_stats,
    detect_paper_key,
    embed_texts,
    evaluate_retrieval,
    find_test_pdfs,
    get_queries_for_paper,
    in_memory_search,
    load_fastembed_model,
    load_structure,
    timer,
)

# Import pipeline internals for custom chunking
from chunking_pipeline import Chunk, ChunkingPipeline
from langchain_text_splitters import RecursiveCharacterTextSplitter
from semantic_splitter import SemanticSplitter

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# ── Chunking strategies ──────────────────────────────────────────────

def strategy_current_pipeline(structure, doc_name: str) -> list[Chunk]:
    """Strategy 1: Current production pipeline (semantic + section-aware)."""
    return chunk_document(structure, doc_name, enrichment_mode="template")


def strategy_fixed_size(structure, doc_name: str, chunk_size: int, overlap_pct: float) -> list[Chunk]:
    """Fixed-size character splitting with overlap."""
    overlap = int(chunk_size * overlap_pct)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " "],
    )

    full_text = _extract_full_text(structure)
    raw_chunks = splitter.split_text(full_text)

    return [
        Chunk(
            text=text,
            metadata={"chunk_type": "child", "page_number": 0, "section": ""},
        )
        for text in raw_chunks
        if len(text.strip()) >= 50
    ]


def strategy_recursive_character(structure, doc_name: str) -> list[Chunk]:
    """Langchain recursive character splitter with default separators."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", ", ", " ", ""],
    )

    full_text = _extract_full_text(structure)
    raw_chunks = splitter.split_text(full_text)

    return [
        Chunk(
            text=text,
            metadata={"chunk_type": "child", "page_number": 0, "section": ""},
        )
        for text in raw_chunks
        if len(text.strip()) >= 50
    ]


def strategy_section_aware(structure, doc_name: str) -> list[Chunk]:
    """One chunk per section — no sub-splitting, respects document structure."""
    chunks = []
    for page in structure.pages:
        for section in page.sections:
            section_type = getattr(section, "section_type", "text")
            if section_type == "image":
                continue

            content = _normalize_text(section.content)
            if len(content) < 50:
                continue

            title = section.title or f"Page {page.page_number}"
            prefix = f"[{title}] " if title not in ("[body]", "") else ""

            chunks.append(Chunk(
                text=prefix + content,
                metadata={
                    "chunk_type": "child",
                    "page_number": page.page_number,
                    "section": title,
                },
                lines=section.lines,
            ))

    return chunks


def _extract_full_text(structure) -> str:
    """Concatenate all text from a document structure."""
    parts = []
    for page in structure.pages:
        for section in page.sections:
            if getattr(section, "section_type", "text") != "image":
                parts.append(section.content or "")
    return "\n\n".join(parts)


def _normalize_text(text: str) -> str:
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ── Evaluation ───────────────────────────────────────────────────────

def evaluate_strategy(
    name: str,
    chunks: list[Chunk],
    model,
    queries: list[tuple[str, list[str], str]],
) -> dict:
    """Evaluate one chunking strategy on quality and retrieval."""
    print(f"\n--- {name} ---")

    chunk_stats = compute_chunk_stats(chunks)
    print(f"  Chunks: {chunk_stats['total_chunks']}, "
          f"avg={chunk_stats['avg_length_chars']} chars, "
          f"mid-sentence breaks={chunk_stats['mid_sentence_break_pct']}%")

    # Embed child chunks only
    child_chunks = [c for c in chunks if c.metadata.get("chunk_type") == "child"]
    if not child_chunks:
        child_chunks = chunks  # strategies without parent/child distinction

    child_texts = [c.text for c in child_chunks]

    with timer("embedding") as t_emb:
        embeddings = embed_texts(model, child_texts)

    emb_stats = compute_embedding_stats(embeddings, t_emb["elapsed_ms"])

    # Retrieval evaluation with section-based ground truth
    results_per_query = []
    query_details = []

    for query, expected_sections, _paper in queries:
        q_emb = embed_texts(model, [query])[0]
        results = in_memory_search(q_emb, embeddings, child_chunks, top_k=5)
        results_per_query.append(results)

        from experiment_harness import section_relevance
        relevant_count = sum(1 for chunk, _ in results if section_relevance(chunk, expected_sections))
        query_details.append({
            "query": query,
            "expected_sections": expected_sections,
            "top_score": round(results[0][1], 4) if results else 0,
            "top_section": results[0][0].metadata.get("section", "") if results else "",
            "top_preview": results[0][0].text[:100] if results else "",
            "relevant_in_top5": relevant_count,
        })

    retrieval = evaluate_retrieval(results_per_query, queries)
    print(f"  MRR={retrieval['mrr']}, P@5={retrieval['precision_at_5']}, NDCG@5={retrieval['ndcg_at_5']}")

    return {
        "strategy": name,
        "chunk_stats": chunk_stats,
        "embedding_stats": emb_stats,
        "retrieval": retrieval,
        "query_details": query_details,
    }


def main():
    parser = argparse.ArgumentParser(description="Chunking strategy comparison experiment")
    parser.add_argument("--pdf", type=str, help="Path to a specific PDF to test with")
    args = parser.parse_args()

    print("=" * 60)
    print("Experiment #5: Chunking Strategy Comparison")
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

    # Extract structure (shared across all strategies)
    with timer("structure_extraction") as t_struct:
        structure = load_structure(pdf_path)
    print(f"Structure: {len(structure.pages)} pages, type={structure.doc_type}")

    # Load embedding model once (shared)
    print(f"\nLoading embedding model: {EMBEDDING_MODEL}")
    with timer("model_load") as t_model:
        model = load_fastembed_model(EMBEDDING_MODEL)

    # Define strategies
    strategies = [
        ("current_pipeline", lambda: strategy_current_pipeline(structure, pdf_path.name)),
        ("fixed_256_10pct", lambda: strategy_fixed_size(structure, pdf_path.name, 256, 0.10)),
        ("fixed_512_15pct", lambda: strategy_fixed_size(structure, pdf_path.name, 512, 0.15)),
        ("fixed_1024_20pct", lambda: strategy_fixed_size(structure, pdf_path.name, 1024, 0.20)),
        ("recursive_character", lambda: strategy_recursive_character(structure, pdf_path.name)),
        ("section_aware", lambda: strategy_section_aware(structure, pdf_path.name)),
    ]

    # Evaluate each strategy
    strategy_results = []
    for name, build_fn in strategies:
        with timer(f"chunk_{name}") as t_chunk:
            chunks = build_fn()
        result = evaluate_strategy(name, chunks, model, queries)
        result["chunking_time_ms"] = t_chunk["elapsed_ms"]
        strategy_results.append(result)

    # Build summary
    valid = [r for r in strategy_results if r["chunk_stats"]["total_chunks"] > 0]
    summary = {}
    if valid:
        best_mrr = max(valid, key=lambda r: r["retrieval"]["mrr"])
        fewest_breaks = min(valid, key=lambda r: r["chunk_stats"]["mid_sentence_break_pct"])
        best_balance = max(valid, key=lambda r: (
            r["retrieval"]["mrr"] * 0.6
            + (1 - r["chunk_stats"]["mid_sentence_break_pct"] / 100) * 0.4
        ))

        summary = {
            "best_retrieval": best_mrr["strategy"],
            "best_boundary_quality": fewest_breaks["strategy"],
            "best_balanced": best_balance["strategy"],
        }

    # Save
    experiment = ExperimentResult(
        experiment_name="chunking_strategy_comparison",
        config={
            "test_document": pdf_path.name,
            "paper_key": paper_key,
            "embedding_model": EMBEDDING_MODEL,
            "strategies_tested": [s[0] for s in strategies],
            "num_queries": len(queries),
            "ground_truth": "section-based (curated expected sections per query)",
        },
        metrics={"summary": summary},
        timings={
            "structure_extraction_ms": t_struct["elapsed_ms"],
            "model_load_ms": t_model["elapsed_ms"],
        },
        details=strategy_results,
    )

    output = experiment.save()

    # Print comparison table
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    print(f"  {'Strategy':25s} {'Chunks':>7s} {'AvgLen':>7s} {'Breaks%':>8s} {'MRR':>6s} {'P@5':>6s}")
    print(f"  {'-'*25} {'-'*7} {'-'*7} {'-'*8} {'-'*6} {'-'*6}")
    for r in strategy_results:
        s = r["chunk_stats"]
        ret = r["retrieval"]
        print(f"  {r['strategy']:25s} {s['total_chunks']:7d} {s['avg_length_chars']:7.0f} "
              f"{s['mid_sentence_break_pct']:7.1f}% {ret['mrr']:6.3f} {ret['precision_at_5']:6.3f}")

    if summary:
        print(f"\n  Best balanced: {summary['best_balanced']}")

    return experiment.to_dict()


if __name__ == "__main__":
    main()
