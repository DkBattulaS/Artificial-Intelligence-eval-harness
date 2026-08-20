"""
Unit tests for the lexical evaluator.

Tests cover:
  - Exact match
  - Partial overlap (paraphrase)
  - No overlap (unrelated answer)
  - Empty answer
  - Empty reference
  - Stopword-only text
  - Expected keywords integration
  - Edge cases: Unicode, numbers, punctuation
  - F1 formula correctness
"""

import pytest
from app.evaluators.lexical import (
    extract_keywords,
    keyword_precision,
    keyword_recall,
    keyword_f1,
    LexicalEvaluator,
)
from app.evaluators.base import EvaluationCase


# ---------------------------------------------------------------------------
# extract_keywords
# ---------------------------------------------------------------------------
class TestExtractKeywords:
    def test_basic_extraction(self):
        kw = extract_keywords("Machine learning uses neural networks")
        assert "machine" in kw
        assert "learning" in kw
        assert "neural" in kw
        assert "networks" in kw

    def test_stopwords_removed(self):
        kw = extract_keywords("the cat is on the mat")
        assert "the" not in kw
        assert "is" not in kw
        assert "on" not in kw
        assert "cat" in kw
        assert "mat" in kw

    def test_short_words_removed(self):
        kw = extract_keywords("AI is an ML concept")
        # "AI", "an" are short (<3 chars); "ML" also short
        assert "ai" not in kw
        assert "ml" not in kw

    def test_empty_string(self):
        assert extract_keywords("") == frozenset()

    def test_whitespace_only(self):
        assert extract_keywords("   ") == frozenset()

    def test_none_like_empty(self):
        # None is not a str — the function handles empty string
        assert extract_keywords("") == frozenset()

    def test_punctuation_stripped(self):
        kw = extract_keywords("deep learning, neural networks!")
        assert "learning" in kw
        assert "neural" in kw
        assert "networks" in kw

    def test_unicode_text(self):
        # Non-ASCII characters should not crash; they'll be filtered by [a-z]+
        kw = extract_keywords("deep learning über alles naïve approach")
        assert "deep" in kw
        assert "learning" in kw
        # "über" and "naïve" contain non-ASCII — filtered out gracefully
        assert "ber" not in kw  # partial not added

    def test_case_insensitive(self):
        kw1 = extract_keywords("Neural Networks")
        kw2 = extract_keywords("neural networks")
        assert kw1 == kw2

    def test_numbers_filtered(self):
        kw = extract_keywords("layer 123 activation function")
        assert "123" not in kw
        assert "layer" in kw
        assert "activation" in kw
        assert "function" in kw


# ---------------------------------------------------------------------------
# keyword_precision
# ---------------------------------------------------------------------------
class TestKeywordPrecision:
    def test_full_overlap(self):
        ans = frozenset({"learning", "data", "model"})
        ref = frozenset({"learning", "data", "model", "training"})
        assert keyword_precision(ans, ref) == 1.0

    def test_partial_overlap(self):
        ans = frozenset({"learning", "data", "noise"})
        ref = frozenset({"learning", "data", "model"})
        # 2/3 overlap
        assert keyword_precision(ans, ref) == round(2 / 3, 4)

    def test_no_overlap(self):
        ans = frozenset({"noise", "random", "unrelated"})
        ref = frozenset({"learning", "data", "model"})
        assert keyword_precision(ans, ref) == 0.0

    def test_empty_answer(self):
        # Undefined when answer has no keywords
        assert keyword_precision(frozenset(), frozenset({"learning"})) is None

    def test_empty_reference(self):
        # 0/1 overlap → 0.0 (all answer terms missing from reference)
        ans = frozenset({"learning"})
        assert keyword_precision(ans, frozenset()) == 0.0

    def test_both_empty(self):
        assert keyword_precision(frozenset(), frozenset()) is None


# ---------------------------------------------------------------------------
# keyword_recall
# ---------------------------------------------------------------------------
class TestKeywordRecall:
    def test_full_recall(self):
        ans = frozenset({"learning", "data", "model", "extra"})
        ref = frozenset({"learning", "data", "model"})
        assert keyword_recall(ans, ref) == 1.0

    def test_partial_recall(self):
        ans = frozenset({"learning", "data"})
        ref = frozenset({"learning", "data", "model"})
        assert keyword_recall(ans, ref) == round(2 / 3, 4)

    def test_no_recall(self):
        ans = frozenset({"noise"})
        ref = frozenset({"learning", "data", "model"})
        assert keyword_recall(ans, ref) == 0.0

    def test_empty_reference(self):
        # Undefined when reference has no keywords
        assert keyword_recall(frozenset({"learning"}), frozenset()) is None

    def test_empty_answer(self):
        # 0/3 overlap → 0.0
        assert keyword_recall(frozenset(), frozenset({"learning", "data", "model"})) == 0.0


# ---------------------------------------------------------------------------
# keyword_f1
# ---------------------------------------------------------------------------
class TestKeywordF1:
    def test_perfect(self):
        assert keyword_f1(1.0, 1.0) == 1.0

    def test_zero_both(self):
        assert keyword_f1(0.0, 0.0) == 0.0

    def test_zero_sum(self):
        # Precision=0, Recall=0 → F1=0 (not undefined)
        assert keyword_f1(0.0, 0.0) == 0.0

    def test_harmonic_mean(self):
        # F1 of 0.5 and 1.0 = 2*(0.5*1.0)/(0.5+1.0) = 0.6667
        result = keyword_f1(0.5, 1.0)
        assert abs(result - round(2 * 0.5 / 1.5, 4)) < 1e-9

    def test_none_precision(self):
        assert keyword_f1(None, 0.8) is None

    def test_none_recall(self):
        assert keyword_f1(0.8, None) is None

    def test_both_none(self):
        assert keyword_f1(None, None) is None


# ---------------------------------------------------------------------------
# LexicalEvaluator (integration)
# ---------------------------------------------------------------------------
class TestLexicalEvaluator:
    def _case(self, answer, reference, expected_keywords=None):
        return EvaluationCase(
            question="test",
            answer=answer,
            reference=reference,
            expected_keywords=expected_keywords,
        )

    def test_exact_match(self):
        evaluator = LexicalEvaluator()
        case = self._case(
            answer="Machine learning uses data to learn patterns",
            reference="Machine learning uses data to learn patterns",
        )
        result = evaluator.evaluate(case)
        assert result["status"] == "ok"
        assert result["metadata"]["keyword_f1"] == 1.0

    def test_paraphrase_has_partial_overlap(self):
        evaluator = LexicalEvaluator()
        case = self._case(
            answer="Neural nets process information through layers",
            reference="Neural networks consist of connected nodes in layers",
        )
        result = evaluator.evaluate(case)
        assert result["status"] == "ok"
        f1 = result["metadata"]["keyword_f1"]
        assert f1 is not None
        assert 0.0 < f1 < 1.0

    def test_unrelated_answer(self):
        evaluator = LexicalEvaluator()
        case = self._case(
            answer="The weather today is sunny and warm",
            reference="Machine learning learns patterns from data",
        )
        result = evaluator.evaluate(case)
        assert result["status"] == "ok"
        f1 = result["metadata"]["keyword_f1"]
        assert f1 == 0.0 or f1 is None  # near-zero overlap

    def test_empty_answer(self):
        evaluator = LexicalEvaluator()
        case = self._case(answer="", reference="Machine learning")
        result = evaluator.evaluate(case)
        assert result["status"] == "ok"
        # answer has no keywords → precision undefined
        assert result["metadata"]["keyword_precision"] is None

    def test_expected_keywords_boost_reference(self):
        evaluator = LexicalEvaluator()
        case = self._case(
            answer="Neural networks have multiple layers",
            reference="Deep learning models",
            expected_keywords=["neural", "layers", "networks"],
        )
        result = evaluator.evaluate(case)
        assert result["status"] == "ok"
        ref_kw = set(result["metadata"]["reference_keywords"])
        assert "neural" in ref_kw
        assert "layers" in ref_kw

    def test_contradictory_answer(self):
        """
        A contradiction with different vocabulary should score low on F1
        (demonstrates that lexical F1 is NOT hallucination detection).
        """
        evaluator = LexicalEvaluator()
        case = self._case(
            answer="Overfitting always improves generalization performance",
            reference="Overfitting reduces a model's ability to generalize",
        )
        result = evaluator.evaluate(case)
        assert result["status"] == "ok"
        # Some overlap ("overfitting", "generali*") — not zero, not one
        f1 = result["metadata"]["keyword_f1"]
        assert f1 is not None
        # This score would LOOK decent despite being wrong — that's the limitation
        # of lexical overlap, correctly NOT called hallucination detection

    def test_unicode_does_not_crash(self):
        evaluator = LexicalEvaluator()
        case = self._case(
            answer="Das Modell lernt Muster aus Daten",
            reference="Machine learning learns patterns from data",
        )
        result = evaluator.evaluate(case)
        assert result["status"] == "ok"

    def test_long_text(self):
        evaluator = LexicalEvaluator()
        answer = " ".join(["learning"] * 500)
        reference = " ".join(["learning", "data", "model"] * 100)
        case = self._case(answer=answer, reference=reference)
        result = evaluator.evaluate(case)
        assert result["status"] == "ok"
        assert result["metadata"]["keyword_f1"] is not None
