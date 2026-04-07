"""Base experiment harness for ML experiments on the EDMS RAG pipeline.

Provides shared utilities for all experiment scripts:
- PDF loading and structure extraction
- Timing and metrics collection
- In-memory embedding search (no database needed)
- Chunk statistics computation
- JSON output
"""

import json
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

# Add smart-search source to import path
SMART_SEARCH_DIR = Path(__file__).resolve().parent.parent / "src" / "smart-search"
sys.path.insert(0, str(SMART_SEARCH_DIR))

from pdf_processor import DocumentStructure, PDFProcessor
from pymupdf4llm_parser import PyMuPDF4LLMParser
from chunking_pipeline import Chunk, ChunkingPipeline
from semantic_splitter import SemanticSplitter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = Path(__file__).resolve().parent


# ── Test data discovery ──────────────────────────────────────────────

def find_test_pdfs() -> list[Path]:
    """Find all PDF files in the project for testing."""
    search_dirs = [
        PROJECT_ROOT / "docs",
        PROJECT_ROOT / "uploads",
        PROJECT_ROOT / "tests" / "fixtures",
        PROJECT_ROOT,
    ]
    pdfs = []
    for d in search_dirs:
        if d.is_dir():
            pdfs.extend(d.glob("*.pdf"))
    return sorted(set(pdfs))


def load_structure(pdf_path: str | Path) -> DocumentStructure:
    """Extract document structure from a PDF using pymupdf4llm."""
    parser = PyMuPDF4LLMParser()
    return parser.extract_structure(str(pdf_path))


def chunk_document(
    structure: DocumentStructure,
    document_name: str,
    embedding_model=None,
    enrichment_mode: str = "template",
) -> list[Chunk]:
    """Run the chunking pipeline on extracted structure."""
    pipeline = ChunkingPipeline(
        embedding_model=embedding_model,
        enrichment_mode=enrichment_mode,
    )
    return pipeline.chunk_document(structure, document_name)


# ── Test PDFs ────────────────────────────────────────────────────────

# Known benchmark papers in the project root
TEST_PDFS = {
    "transformer": "1706.03762v7.pdf",        # Attention Is All You Need
    "deepseekmath": "2402.03300v3.pdf",        # DeepSeekMath 7B
}

# Existing chunk data for baseline comparison (from data-analyst task #2)
BASELINE_CHUNKS = {
    "deepseekmath_default": "docs/deepseekmath_chunks.json",
    "deepseekmath_pymupdf4llm": "docs/deepseekmath_chunks_pymupdf4llm.json",
    "deepseekmath_docling": "docs/deepseekmath_chunks_docling.json",
    "deepseekmath_unstructured": "docs/deepseekmath_chunks_unstructured.json",
}


def load_baseline_chunks(key: str) -> list[dict]:
    """Load pre-existing chunk data from JSON for comparison."""
    rel_path = BASELINE_CHUNKS.get(key)
    if not rel_path:
        return []
    full_path = PROJECT_ROOT / rel_path
    if not full_path.exists():
        return []
    with open(full_path) as f:
        return json.load(f)


# ── Curated test queries with ground-truth sections ──────────────────
# Each entry: (query, expected_sections, paper)
# expected_sections = list of section title substrings that count as relevant

TRANSFORMER_QUERIES = [
    ("What is the Transformer architecture?",
     ["abstract", "model architecture", "encoder", "decoder"],
     "transformer"),
    ("How does multi-head attention work?",
     ["multi-head attention", "3.2", "attention"],
     "transformer"),
    ("What BLEU scores did the Transformer achieve?",
     ["abstract", "results", "training", "6.1", "bleu"],
     "transformer"),
    ("What is scaled dot-product attention?",
     ["scaled dot-product", "3.2.1", "attention"],
     "transformer"),
    ("How does positional encoding work in the Transformer?",
     ["positional encoding", "3.5", "positional"],
     "transformer"),
    ("Compare self-attention computational complexity to recurrence",
     ["table 1", "complexity", "self-attention", "4", "why self-attention"],
     "transformer"),
]

DEEPSEEKMATH_QUERIES = [
    ("What is GRPO and how does it differ from PPO?",
     ["grpo", "group relative policy", "reinforcement learning", "3.4"],
     "deepseekmath"),
    ("How was the DeepSeekMath training corpus created?",
     ["data collection", "data selection", "common crawl", "2", "math corpus"],
     "deepseekmath"),
    ("What benchmarks does DeepSeekMath 7B achieve on MATH?",
     ["evaluation", "results", "math benchmark", "51.7"],
     "deepseekmath"),
    ("Why start from DeepSeek-Coder-Base instead of a general LLM?",
     ["introduction", "1", "coder-base", "base model"],
     "deepseekmath"),
    ("What is the fastText data selection pipeline?",
     ["data selection", "fasttext", "2.1", "pipeline"],
     "deepseekmath"),
]

# Combined flat list (backward-compatible with scripts that just need query strings)
STANDARD_QUERIES = [q for q, _, _ in TRANSFORMER_QUERIES + DEEPSEEKMATH_QUERIES]

# Full curated query set with ground truth
CURATED_QUERIES = TRANSFORMER_QUERIES + DEEPSEEKMATH_QUERIES


def get_queries_for_paper(paper_key: str) -> list[tuple[str, list[str], str]]:
    """Return curated queries for a specific paper."""
    if paper_key == "transformer":
        return TRANSFORMER_QUERIES
    elif paper_key == "deepseekmath":
        return DEEPSEEKMATH_QUERIES
    return CURATED_QUERIES


# ── Timing ───────────────────────────────────────────────────────────

@contextmanager
def timer(label: str = ""):
    """Context manager that measures wall-clock time in milliseconds."""
    start = time.perf_counter()
    result = {"elapsed_ms": 0.0}
    yield result
    result["elapsed_ms"] = round((time.perf_counter() - start) * 1000, 2)
    if label:
        print(f"  [{label}] {result['elapsed_ms']:.1f}ms")


# ── Experiment result container ──────────────────────────────────────

@dataclass
class ExperimentResult:
    """Collects config, metrics, and timing for one experiment run."""

    experiment_name: str
    config: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    timings: dict = field(default_factory=dict)
    details: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "experiment": self.experiment_name,
            "config": self.config,
            "metrics": self.metrics,
            "timings": self.timings,
            "details": self.details,
        }

    def save(self, output_path: str | Path | None = None) -> Path:
        if output_path is None:
            output_path = SCRIPTS_DIR / f"{self.experiment_name}_results.json"
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        print(f"\nResults saved to {output_path}")
        return output_path


# ── Chunk statistics ─────────────────────────────────────────────────

def compute_chunk_stats(chunks: list[Chunk]) -> dict:
    """Compute descriptive statistics over a set of chunks."""
    if not chunks:
        return {"count": 0}

    lengths = [len(c.text) for c in chunks]
    child_chunks = [c for c in chunks if c.metadata.get("chunk_type") == "child"]
    parent_chunks = [c for c in chunks if c.metadata.get("chunk_type") == "parent"]
    child_lengths = [len(c.text) for c in child_chunks]

    # Mid-sentence break detection
    sentence_breaks = sum(
        1 for c in chunks
        if c.text.rstrip() and c.text.rstrip()[-1] not in '.!?:;"\')]\n'
    )

    stats = {
        "total_chunks": len(chunks),
        "parent_chunks": len(parent_chunks),
        "child_chunks": len(child_chunks),
        "avg_length_chars": round(sum(lengths) / len(lengths), 1),
        "min_length_chars": min(lengths),
        "max_length_chars": max(lengths),
        "median_length_chars": round(sorted(lengths)[len(lengths) // 2], 1),
        "mid_sentence_breaks": sentence_breaks,
        "mid_sentence_break_pct": round(sentence_breaks / len(chunks) * 100, 1),
    }

    if child_lengths:
        stats["child_avg_length"] = round(sum(child_lengths) / len(child_lengths), 1)
        stats["child_min_length"] = min(child_lengths)
        stats["child_max_length"] = max(child_lengths)

    return stats


# ── Embedding utilities ──────────────────────────────────────────────

def load_fastembed_model(model_name: str):
    """Load a fastembed TextEmbedding model by name."""
    from fastembed import TextEmbedding
    return TextEmbedding(model_name)


def embed_texts(model, texts: list[str]) -> list[list[float]]:
    """Embed a list of texts, returning list of vectors."""
    return [emb.tolist() for emb in model.embed(texts)]


def compute_embedding_stats(embeddings: list, elapsed_ms: float) -> dict:
    """Compute statistics about a batch of embeddings."""
    import numpy as np

    if not embeddings:
        return {}

    arr = np.array(embeddings)
    norms = np.linalg.norm(arr, axis=1)
    return {
        "count": len(embeddings),
        "dimensions": arr.shape[1],
        "total_time_ms": round(elapsed_ms, 2),
        "per_embedding_ms": round(elapsed_ms / len(embeddings), 2),
        "embeddings_per_sec": round(len(embeddings) / (elapsed_ms / 1000), 1) if elapsed_ms > 0 else 0,
        "norm_mean": round(float(np.mean(norms)), 4),
        "norm_std": round(float(np.std(norms)), 4),
    }


# ── In-memory vector search (no database needed) ────────────────────

def cosine_similarity(a, b) -> float:
    """Cosine similarity between two vectors."""
    import numpy as np

    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom else 0.0


def in_memory_search(
    query_embedding: list[float],
    chunk_embeddings: list[list[float]],
    chunks: list[Chunk],
    top_k: int = 5,
) -> list[tuple[Chunk, float]]:
    """Cosine-similarity search over in-memory embeddings."""
    scores = [
        (i, cosine_similarity(query_embedding, emb))
        for i, emb in enumerate(chunk_embeddings)
    ]
    scores.sort(key=lambda x: x[1], reverse=True)
    return [(chunks[i], score) for i, score in scores[:top_k]]


# ── Retrieval quality metrics ────────────────────────────────────────

def mean_reciprocal_rank(results_per_query: list[list[bool]]) -> float:
    """MRR: average of 1/rank of the first relevant result per query."""
    mrrs = []
    for relevance_list in results_per_query:
        for rank, is_relevant in enumerate(relevance_list, 1):
            if is_relevant:
                mrrs.append(1.0 / rank)
                break
        else:
            mrrs.append(0.0)
    return round(sum(mrrs) / len(mrrs), 4) if mrrs else 0.0


def precision_at_k(results_per_query: list[list[bool]], k: int = 5) -> float:
    """Average P@k across queries."""
    precisions = []
    for relevance_list in results_per_query:
        top_k = relevance_list[:k]
        precisions.append(sum(top_k) / k if top_k else 0.0)
    return round(sum(precisions) / len(precisions), 4) if precisions else 0.0


def ndcg_at_k(results_per_query: list[list[float]], k: int = 5) -> float:
    """Average NDCG@k across queries. Scores are graded relevance [0, 1]."""
    import math

    def _dcg(scores, k):
        return sum(s / math.log2(i + 2) for i, s in enumerate(scores[:k]))

    ndcgs = []
    for scores in results_per_query:
        dcg = _dcg(scores, k)
        ideal = _dcg(sorted(scores, reverse=True), k)
        ndcgs.append(dcg / ideal if ideal > 0 else 0.0)
    return round(sum(ndcgs) / len(ndcgs), 4) if ndcgs else 0.0


# ── Relevance judgement ──────────────────────────────────────────────

def section_relevance(chunk, expected_sections: list[str]) -> bool:
    """Check if a chunk comes from (or mentions) one of the expected sections.

    Uses both chunk metadata (section field) and content matching against
    the curated expected_sections list.  This is much more reliable than
    pure keyword overlap for papers with known structure.
    """
    chunk_section = (chunk.metadata.get("section", "") if hasattr(chunk, "metadata") else "").lower()
    chunk_text = (chunk.text if hasattr(chunk, "text") else str(chunk)).lower()

    for expected in expected_sections:
        expected_lower = expected.lower()
        if expected_lower in chunk_section:
            return True
        if expected_lower in chunk_text:
            return True
    return False


def graded_relevance(chunk, expected_sections: list[str]) -> float:
    """Graded relevance score (0.0–1.0) for NDCG computation.

    1.0 = section metadata matches expected section
    0.5 = content text mentions expected section keywords
    0.0 = no match
    """
    chunk_section = (chunk.metadata.get("section", "") if hasattr(chunk, "metadata") else "").lower()
    chunk_text = (chunk.text if hasattr(chunk, "text") else str(chunk)).lower()

    for expected in expected_sections:
        expected_lower = expected.lower()
        if expected_lower in chunk_section:
            return 1.0

    for expected in expected_sections:
        expected_lower = expected.lower()
        if expected_lower in chunk_text:
            return 0.5

    return 0.0


def keyword_relevance_score(query: str, chunk_text: str) -> float:
    """Simple keyword overlap relevance score (0-1) for automated evaluation."""
    query_terms = set(query.lower().split())
    chunk_lower = chunk_text.lower()
    matched = sum(1 for t in query_terms if t in chunk_lower)
    return matched / len(query_terms) if query_terms else 0.0


def evaluate_retrieval(
    results_per_query: list[list[tuple]],
    queries_with_ground_truth: list[tuple[str, list[str], str]],
) -> dict:
    """Compute retrieval metrics using section-based ground truth.

    Args:
        results_per_query: For each query, list of (chunk, score) tuples.
        queries_with_ground_truth: Curated queries with expected sections.

    Returns dict with MRR, P@5, NDCG@5.
    """
    binary_per_query = []
    graded_per_query = []

    for (query, expected_sections, _paper), results in zip(
        queries_with_ground_truth, results_per_query
    ):
        binary = [section_relevance(chunk, expected_sections) for chunk, _ in results]
        graded = [graded_relevance(chunk, expected_sections) for chunk, _ in results]
        binary_per_query.append(binary)
        graded_per_query.append(graded)

    return {
        "mrr": mean_reciprocal_rank(binary_per_query),
        "precision_at_5": precision_at_k(binary_per_query, k=5),
        "ndcg_at_5": ndcg_at_k(graded_per_query, k=5),
        "queries_evaluated": len(queries_with_ground_truth),
    }


# ── Main ─────────────────────────────────────────────────────────────

def detect_paper_key(pdf_path: Path) -> str | None:
    """Detect which curated paper a PDF corresponds to."""
    name = pdf_path.name.lower()
    if "1706.03762" in name:
        return "transformer"
    if "2402.03300" in name:
        return "deepseekmath"
    return None


if __name__ == "__main__":
    print("=== EDMS Experiment Harness ===\n")

    pdfs = find_test_pdfs()
    print(f"Found {len(pdfs)} test PDF(s):")
    for p in pdfs:
        paper_key = detect_paper_key(p)
        label = f" [{paper_key}]" if paper_key else ""
        print(f"  - {p.name} ({p.stat().st_size // 1024}KB){label}")

    print(f"\nCurated queries: {len(CURATED_QUERIES)}")
    print(f"  Transformer: {len(TRANSFORMER_QUERIES)} queries")
    print(f"  DeepSeekMath: {len(DEEPSEEKMATH_QUERIES)} queries")

    print(f"\nBaseline chunk data:")
    for key, path in BASELINE_CHUNKS.items():
        full = PROJECT_ROOT / path
        status = f"{len(load_baseline_chunks(key))} chunks" if full.exists() else "missing"
        print(f"  - {key}: {status}")

    if pdfs:
        pdf = pdfs[0]
        print(f"\nLoading structure from {pdf.name}...")
        with timer("structure_extraction") as t:
            structure = load_structure(pdf)
        print(f"  Doc type: {structure.doc_type}")
        print(f"  Pages: {len(structure.pages)}")
        print(f"  Title: {structure.title}")

        print(f"\nChunking...")
        with timer("chunking") as t:
            chunks = chunk_document(structure, pdf.name)
        stats = compute_chunk_stats(chunks)
        print(f"  Chunks: {stats['total_chunks']} ({stats['parent_chunks']} parents, {stats['child_chunks']} children)")
        print(f"  Avg length: {stats['avg_length_chars']} chars")
        print(f"  Mid-sentence breaks: {stats['mid_sentence_break_pct']}%")

        print("\nHarness ready for experiments.")
