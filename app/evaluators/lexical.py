"""
Lexical evaluator: keyword precision, recall, and F1.

What these measure:
  keyword_precision : fraction of answer keywords found in the reference
  keyword_recall    : fraction of reference keywords found in the answer
  keyword_f1        : harmonic mean of the above two

What these do NOT measure:
  - Semantic correctness (a paraphrase scores low; an exact copy scores high)
  - Hallucination (a fluent wrong answer with different vocabulary scores low
    even if every claim is supported)
  - Meaning overlap (only surface-form token overlap)

These are simple, fast, and interpretable as lexical overlap metrics.
They are most useful as a complement to semantic similarity.

Keyword extraction:
  - Lowercased alphabetic tokens only
  - Stop words removed (configurable)
  - Minimum token length: 3 characters
  - Optionally supplemented with dataset-provided expected_keywords
"""

from __future__ import annotations

import re
from typing import Optional

_STOPWORDS: frozenset[str] = frozenset({
    "is", "a", "the", "of", "and", "to", "in", "that", "it", "an", "are",
    "for", "on", "with", "as", "by", "this", "was", "be", "or", "at", "from",
    "but", "not", "have", "had", "has", "its", "we", "they", "you", "he",
    "she", "do", "did", "will", "would", "could", "should", "may", "might",
    "can", "into", "more", "also", "than", "when", "where", "which", "who",
    "so", "if", "up", "out", "about", "what", "all", "been",
})


def extract_keywords(text: str) -> frozenset[str]:
    """
    Extract meaningful keywords from text.
    Returns empty frozenset for empty/None input.
    """
    if not text:
        return frozenset()
    words = re.findall(r"\b[a-z]+\b", text.strip().lower())
    return frozenset(w for w in words if w not in _STOPWORDS and len(w) >= 3)


def keyword_precision(answer_kw: frozenset[str], reference_kw: frozenset[str]) -> Optional[float]:
    """Fraction of answer keywords in reference. None if answer has no keywords."""
    if not answer_kw:
        return None
    return round(len(answer_kw & reference_kw) / len(answer_kw), 4)


def keyword_recall(answer_kw: frozenset[str], reference_kw: frozenset[str]) -> Optional[float]:
    """Fraction of reference keywords in answer. None if reference has no keywords."""
    if not reference_kw:
        return None
    return round(len(answer_kw & reference_kw) / len(reference_kw), 4)


def keyword_f1(
    precision: Optional[float], recall: Optional[float]
) -> Optional[float]:
    """Harmonic mean. None if either component is None."""
    if precision is None or recall is None:
        return None
    denom = precision + recall
    if denom == 0.0:
        return 0.0
    return round(2 * precision * recall / denom, 4)


class LexicalEvaluator:
    """
    Computes keyword precision, recall, and F1 for a single evaluation case.
    Optionally merges dataset-provided expected_keywords into the reference
    keyword set.
    """

    name = "lexical"

    def evaluate(self, case) -> dict:
        """
        Returns a dict with keys:
          keyword_precision, keyword_recall, keyword_f1
        Each is a float in [0, 1] or None if undefined.
        """
        try:
            answer_kw = extract_keywords(case.answer)
            reference_kw = extract_keywords(case.reference)

            # Optionally add dataset-level expected keywords to the reference set
            if case.expected_keywords:
                extra = frozenset(
                    w.lower().strip()
                    for kw in case.expected_keywords
                    for w in re.findall(r"\b[a-z]+\b", kw.lower())
                    if len(w) >= 3
                )
                reference_kw = reference_kw | extra

            prec = keyword_precision(answer_kw, reference_kw)
            rec = keyword_recall(answer_kw, reference_kw)
            f1 = keyword_f1(prec, rec)

            return {
                "metric": self.name,
                "score": f1,  # primary score is F1
                "explanation": (
                    f"Keyword overlap — precision: {prec}, recall: {rec}, F1: {f1}. "
                    f"Answer keywords: {len(answer_kw)}, Reference keywords: {len(reference_kw)}."
                ),
                "status": "ok",
                "metadata": {
                    "keyword_precision": prec,
                    "keyword_recall": rec,
                    "keyword_f1": f1,
                    "answer_keywords": sorted(answer_kw),
                    "reference_keywords": sorted(reference_kw),
                },
            }
        except Exception as exc:
            return {
                "metric": self.name,
                "score": None,
                "explanation": str(exc),
                "status": "failed",
                "metadata": {},
            }
