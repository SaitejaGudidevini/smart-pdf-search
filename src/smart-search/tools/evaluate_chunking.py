"""Evaluate chunking quality against synthetic fixture documents.

Run inside the sidecar container or an environment with the chunking
dependencies installed:

    python tools/evaluate_chunking.py
    python tools/evaluate_chunking.py test_fixtures/research_paper_frontmatter.json --strict
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from chunking_pipeline import ChunkingPipeline


@dataclass
class FixtureSection:
    title: str
    level: int
    content: str
    lines: list = field(default_factory=list)
    section_type: str = "text"


@dataclass
class FixturePage:
    page_number: int
    sections: list[FixtureSection]


@dataclass
class FixtureStructure:
    title: str
    doc_type: str
    pages: list[FixturePage]


def build_structure(payload: dict) -> FixtureStructure:
    pages = []
    for page in payload["pages"]:
        sections = [
            FixtureSection(
                title=section.get("title", ""),
                level=section.get("level", 3),
                content=section.get("content", ""),
                lines=section.get("lines", []),
                section_type=section.get("section_type", "text"),
            )
            for section in page.get("sections", [])
        ]
        pages.append(FixturePage(page_number=page["page_number"], sections=sections))
    return FixtureStructure(
        title=payload.get("title", "Fixture"),
        doc_type=payload.get("doc_type", "general"),
        pages=pages,
    )


def section_name(chunk) -> str:
    return (chunk.metadata.get("section") or "").strip()


def duplicate_ratio(parent_text: str, child_text: str) -> float:
    left = " ".join(parent_text.split())
    right = " ".join(child_text.split())
    if not left or not right:
        return 1.0
    shorter = min(len(left), len(right))
    longer = max(len(left), len(right))
    overlap = shorter / longer
    if left == right:
        return 1.0
    if left in right or right in left:
        return overlap
    return 0.0


def evaluate_fixture(path: Path, strict: bool = False) -> tuple[bool, str]:
    payload = json.loads(path.read_text())
    structure = build_structure(payload)
    pipeline = ChunkingPipeline()
    chunks = pipeline.chunk_document(
        structure=structure,
        document_name=payload.get("document_name", path.stem),
        document_key=payload.get("document_key", f"fixture:{path.stem}"),
    )

    parents = [chunk for chunk in chunks if chunk.metadata.get("chunk_type") == "parent"]
    children = [chunk for chunk in chunks if chunk.metadata.get("chunk_type") == "child"]
    expectations = payload.get("expectations", {})

    duplicate_children = 0
    parent_text_by_id = {
        chunk.metadata.get("parent_id"): chunk.text
        for chunk in parents
        if chunk.metadata.get("parent_id")
    }
    for child in children:
        parent_id = child.metadata.get("parent_id")
        parent_text = parent_text_by_id.get(parent_id)
        if parent_text and duplicate_ratio(parent_text, child.text) >= 0.9:
            duplicate_children += 1

    suspicious_titles = [
        section_name(chunk)
        for chunk in parents
        if section_name(chunk)
        and (
            section_name(chunk).startswith("[body]")
            or re.match(r"^[a-z]", section_name(chunk))
            or re.match(r"^[^A-Z0-9]", section_name(chunk))
            or len(section_name(chunk).split()) <= 2 and section_name(chunk).endswith(".")
        )
    ]

    short_parents = [chunk for chunk in parents if len(chunk.text) < expectations.get("min_parent_chars", 140)]
    forbidden_hits = []
    for pattern in expectations.get("forbidden_section_patterns", []):
        regex = re.compile(pattern, re.IGNORECASE)
        forbidden_hits.extend(chunk for chunk in parents if regex.search(section_name(chunk)) or regex.search(chunk.text))

    missing_sections = []
    for expected in expectations.get("must_include_sections", []):
        if not any(expected.lower() in section_name(chunk).lower() for chunk in parents):
            missing_sections.append(expected)

    # Check contextual enrichment: every child should have enriched_text
    unenriched_children = [
        chunk for chunk in children
        if not chunk.metadata.get("enriched_text")
    ]

    # Check table chunks are kept whole (not split)
    table_chunks = [
        chunk for chunk in chunks
        if "[Table" in chunk.text
    ]

    metrics = {
        "parents": len(parents),
        "children": len(children),
        "duplicates": duplicate_children,
        "short_parents": len(short_parents),
        "forbidden_hits": len(forbidden_hits),
        "suspicious_titles": len(suspicious_titles),
        "missing_sections": len(missing_sections),
        "unenriched_children": len(unenriched_children),
        "table_chunks": len(table_chunks),
    }

    failures = []
    if len(parents) > expectations.get("max_parents", 9999):
        failures.append(f"parent count {len(parents)} > {expectations['max_parents']}")
    if len(children) > expectations.get("max_children", 9999):
        failures.append(f"child count {len(children)} > {expectations['max_children']}")
    if duplicate_children > expectations.get("max_duplicate_children", 9999):
        failures.append(
            f"duplicate child chunks {duplicate_children} > {expectations['max_duplicate_children']}"
        )
    if len(short_parents) > expectations.get("max_short_parents", 9999):
        failures.append(f"short parent chunks {len(short_parents)} > {expectations['max_short_parents']}")
    if len(forbidden_hits) > expectations.get("max_forbidden_hits", 9999):
        failures.append(f"forbidden section hits {len(forbidden_hits)} > {expectations['max_forbidden_hits']}")
    if missing_sections:
        failures.append(f"missing sections: {', '.join(missing_sections)}")

    if unenriched_children:
        failures.append(
            f"unenriched children {len(unenriched_children)} (all children should have enriched_text)"
        )

    lines = [
        f"{path.name}",
        f"  parents={metrics['parents']} children={metrics['children']} duplicates={metrics['duplicates']}",
        f"  short_parents={metrics['short_parents']} forbidden_hits={metrics['forbidden_hits']} suspicious_titles={metrics['suspicious_titles']}",
        f"  enriched={metrics['children'] - metrics['unenriched_children']}/{metrics['children']} table_chunks={metrics['table_chunks']}",
    ]
    if suspicious_titles:
        lines.append(f"  suspicious_titles={', '.join(sorted(set(suspicious_titles))[:5])}")
    if failures:
        lines.append("  status=FAIL")
        for item in failures:
            lines.append(f"  - {item}")
    else:
        lines.append("  status=PASS")

    passed = not failures
    if strict and not passed:
        return False, "\n".join(lines)
    return True, "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("fixtures", nargs="*", help="Fixture file paths. Defaults to all JSON fixtures.")
    parser.add_argument("--strict", action="store_true", help="Return non-zero when any fixture fails.")
    args = parser.parse_args()

    base = Path(__file__).resolve().parent.parent / "test_fixtures"
    fixture_paths = [Path(item).resolve() for item in args.fixtures] if args.fixtures else sorted(base.glob("*.json"))
    if not fixture_paths:
        print("No fixtures found.")
        return 1

    overall_ok = True
    for path in fixture_paths:
        ok, report = evaluate_fixture(path=path, strict=args.strict)
        print(report)
        overall_ok = overall_ok and ok

    return 0 if overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
