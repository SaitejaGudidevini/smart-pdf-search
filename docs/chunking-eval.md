# Chunking Evaluation

The RAG chunker now has a small synthetic evaluation corpus so we can
measure quality intentionally instead of inspecting a single PDF by eye.

## Fixtures

Fixtures live in [`src/smart-search/test_fixtures`](/Users/saiteja/Documents/Dev/EDMS/src/smart-search/test_fixtures) and cover:

- `research_paper_frontmatter.json`
  Tests title block and author metadata suppression.
- `research_paper_references.json`
  Tests reference-section exclusion for research papers.
- `layout_noise_and_captions.json`
  Tests image/caption separation and layout-noise handling.

## Evaluator

Use [`src/smart-search/tools/evaluate_chunking.py`](/Users/saiteja/Documents/Dev/EDMS/src/smart-search/tools/evaluate_chunking.py).

Typical usage inside the sidecar container:

```bash
docker exec rag-sidecar python /app/tools/evaluate_chunking.py
docker exec rag-sidecar python /app/tools/evaluate_chunking.py --strict
```

The evaluator reports:

- parent and child counts
- near-duplicate child chunks
- short parent chunks
- forbidden content hits
- suspicious section titles
- missing expected sections

`--strict` returns non-zero if any fixture fails.

## Intent

This corpus is not a full benchmark yet. It is a regression harness for the
chunking logic so we can improve section detection, frontmatter suppression,
reference exclusion, and figure/caption handling without losing previous gains.
