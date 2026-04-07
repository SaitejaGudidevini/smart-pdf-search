"""Extract keywords from raw document text using spaCy.

Produces up to 200 unique, deduplicated keywords per document covering:
1. Named entities (companies, people, places, dates, money)
2. Numbers and amounts ($19,936, 2026, 15%)
3. Domain terms (revenue, liability, monitoring)
4. Proper nouns (Microsoft, Apple, CTSI)

Usage:
    extractor = KeywordExtractor()
    keywords = extractor.extract("full document text here...")
    # Returns: ["Microsoft", "$8,065", "revenue", "2003", "net income", ...]
"""

from __future__ import annotations

import logging
import re
from collections import Counter

logger = logging.getLogger(__name__)

MAX_KEYWORDS = 200

# Common stopwords and noise to skip
_SKIP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "must", "to", "of",
    "in", "for", "on", "with", "at", "by", "from", "as", "into", "through",
    "during", "before", "after", "above", "below", "between", "under",
    "again", "further", "then", "once", "here", "there", "when", "where",
    "why", "how", "all", "each", "every", "both", "few", "more", "most",
    "other", "some", "such", "no", "nor", "not", "only", "own", "same",
    "so", "than", "too", "very", "just", "because", "but", "and", "or",
    "if", "while", "that", "this", "these", "those", "it", "its",
    "he", "she", "they", "we", "you", "i", "me", "him", "her", "us",
    "them", "my", "your", "his", "our", "their", "what", "which", "who",
    "whom", "whose", "also", "however", "therefore", "thus", "hence",
    "page", "section", "table", "figure", "see", "note", "e.g.", "i.e.",
    "etc", "shall", "upon", "herein", "thereof", "hereby", "therein",
    "pursuant", "whereas", "notwithstanding", "hereunder", "hereof",
    # Markdown/formatting noise
    "br", "http", "https", "www", "com", "org", "pdf", "xlsx", "csv",
}

_SKIP_PATTERNS = re.compile(
    r"^[\d.,%$€£¥]+$|"  # pure numbers/symbols already captured separately
    r"^[a-z]{1,2}$|"     # single/double letter words
    r"^\W+$|"            # pure punctuation
    r"^_+$"              # underscores
)


class KeywordExtractor:
    """Extract keywords from document text using spaCy NER + noun extraction."""

    def __init__(self):
        self._nlp = None

    def _load_model(self):
        if self._nlp is None:
            import spacy
            try:
                self._nlp = spacy.load("en_core_web_sm", disable=["parser"])
                logger.info("spaCy model loaded for keyword extraction")
            except OSError:
                import subprocess
                subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
                self._nlp = spacy.load("en_core_web_sm", disable=["parser"])

    def extract(self, text: str, max_keywords: int = MAX_KEYWORDS) -> list[str]:
        """Extract up to max_keywords unique keywords from raw document text.

        Returns a deduplicated list ordered by importance (frequency + entity type).
        """
        self._load_model()

        # Truncate very long docs for spaCy (process first 50K chars + last 10K)
        if len(text) > 60000:
            text = text[:50000] + "\n...\n" + text[-10000:]

        doc = self._nlp(text)

        seen_lower: set[str] = set()
        scored: dict[str, float] = {}  # keyword → importance score

        # 1. Named entities (highest priority)
        entity_weights = {
            "ORG": 5.0,      # companies, organizations
            "PERSON": 4.0,   # people names
            "MONEY": 4.5,    # dollar amounts
            "DATE": 3.0,     # dates
            "GPE": 3.5,      # countries, cities, states
            "PERCENT": 3.0,  # percentages
            "CARDINAL": 2.0, # numbers
            "PRODUCT": 3.5,  # product names
            "LAW": 3.5,      # legal references
            "FAC": 2.5,      # facilities
            "EVENT": 2.5,    # events
            "QUANTITY": 2.0, # quantities
            "ORDINAL": 1.5,  # first, second
        }

        for ent in doc.ents:
            clean = ent.text.strip()
            if len(clean) < 2 or len(clean) > 80:
                continue
            lower = clean.lower()
            if lower in _SKIP_WORDS or lower in seen_lower:
                continue
            if _SKIP_PATTERNS.match(lower):
                continue

            weight = entity_weights.get(ent.label_, 1.0)
            seen_lower.add(lower)
            scored[clean] = scored.get(clean, 0) + weight

        # 2. Money amounts and percentages from text (regex — catches what spaCy misses)
        money_pattern = re.compile(r"\$[\d,]+(?:\.\d+)?")
        for match in money_pattern.finditer(text):
            val = match.group()
            if val.lower() not in seen_lower:
                seen_lower.add(val.lower())
                scored[val] = scored.get(val, 0) + 4.0

        pct_pattern = re.compile(r"\d+(?:\.\d+)?%")
        for match in pct_pattern.finditer(text):
            val = match.group()
            if val.lower() not in seen_lower:
                seen_lower.add(val.lower())
                scored[val] = scored.get(val, 0) + 3.0

        # 3. Years (4-digit numbers between 1900-2100)
        year_pattern = re.compile(r"\b(19|20)\d{2}\b")
        for match in year_pattern.finditer(text):
            val = match.group()
            if val not in seen_lower:
                seen_lower.add(val)
                scored[val] = scored.get(val, 0) + 3.0

        # 4. Noun chunks and domain terms (medium priority)
        # Use POS tags for nouns and proper nouns
        noun_freq: Counter = Counter()
        for token in doc:
            if token.pos_ in ("NOUN", "PROPN") and not token.is_stop:
                lemma = token.lemma_.strip()
                if (
                    len(lemma) >= 3
                    and lemma.lower() not in _SKIP_WORDS
                    and lemma.lower() not in seen_lower
                    and not _SKIP_PATTERNS.match(lemma.lower())
                ):
                    noun_freq[lemma] += 1

        # Add frequent nouns with frequency-based scoring
        for noun, freq in noun_freq.most_common(150):
            lower = noun.lower()
            if lower in seen_lower:
                continue
            # Score: base 1.0 + log frequency bonus
            import math
            score = 1.0 + math.log(freq + 1)
            seen_lower.add(lower)
            scored[noun] = score

        # 5. Sort by score, take top max_keywords
        sorted_keywords = sorted(scored.items(), key=lambda x: x[1], reverse=True)
        keywords = [kw for kw, _ in sorted_keywords[:max_keywords]]

        logger.info("Extracted %d keywords from %d chars of text", len(keywords), len(text))
        return keywords

    def extract_from_chunks(self, chunks: list) -> list[str]:
        """Extract keywords from a list of chunk objects (uses content field)."""
        all_text_parts = []
        for chunk in chunks:
            text = chunk.text if hasattr(chunk, "text") else str(chunk)
            all_text_parts.append(text)

        full_text = "\n".join(all_text_parts)
        return self.extract(full_text)
