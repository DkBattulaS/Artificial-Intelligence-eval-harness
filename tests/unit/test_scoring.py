"""
Unit tests for metric computation helpers.

Tests cover:
  - semantic similarity computation (mocked embeddings)
  - keyword metric edge cases
  - NaN/None handling — no fabrication
  - llm_judge merge into compute_scores
  - Aggregation helpers
"""

import pytest
import torch
from app.services.evaluation_service import (
    compute_semantic_similarity,
    compute_keyword_precision,
    compute_keyword_recall,
    compute_keyword_f1,
    compute_scores,
    normalize,
    extract_keywords,
)
from app.engine.runner import _avg, _aggregate
from app.models.evaluation import CaseResult


# ---------------------------------------------------------------------------
# normalize
# ---------------------------------------------------------------------------
class TestNormalize:
    def test_lowercase(self):
        assert normalize("Hello World") == "hello world"

    def test_collapse_whitespace(self):
        assert normalize("  hello   world  ") == "hello world"

    def test_empty(self):
        assert normalize("") == ""

    def test_newlines(self):
        assert normalize("hello\nworld") == "hello world"


# ---------------------------------------------------------------------------
# extract_keywords (from evaluation_service)
# ---------------------------------------------------------------------------
class TestExtractKeywordsService:
    def test_returns_frozenset(self):
        result = extract_keywords("machine learning")
        assert isinstance(result, frozenset)

    def test_empty_returns_empty_frozenset(self):
        assert extract_keywords("") == frozenset()

    def test_stopwords_filtered(self):
        kw = extract_keywords("the cat and the dog")
        assert "the" not in kw
        assert "and" not in kw
        assert "cat" in kw
        assert "dog" in kw


# ---------------------------------------------------------------------------
# compute_semantic_similarity (with mock tensors)
# ---------------------------------------------------------------------------
class TestComputeSemanticSimilarity:
    def test_identical_embeddings_score_one(self):
        v = torch.tensor([[1.0, 0.0, 0.0]])
        score = compute_semantic_similarity(v, v)
        assert score == 1.0

    def test_orthogonal_embeddings_score_zero(self):
        v1 = torch.tensor([[1.0, 0.0]])
        v2 = torch.tensor([[0.0, 1.0]])
        score = compute_semantic_similarity(v1, v2)
        assert score == 0.0

    def test_score_is_float(self):
        v = torch.tensor([[0.5, 0.5]])
        score = compute_semantic_similarity(v, v)
        assert isinstance(score, float)

    def test_score_rounded_to_4_places(self):
        v1 = torch.tensor([[1.0, 2.0, 3.0]])
        v2 = torch.tensor([[4.0, 5.0, 6.0]])
        score = compute_semantic_similarity(v1, v2)
        assert score == round(score, 4)


# ---------------------------------------------------------------------------
# compute_scores with mocked embeddings
# ---------------------------------------------------------------------------
class TestComputeScores:
    def _embeddings(self, v):
        return torch.tensor([v])

    def test_returns_all_keys(self):
        emb = self._embeddings([1.0, 0.0])
        result = compute_scores("machine learning", "machine learning", emb, emb)
        expected_keys = {
            "semantic_similarity",
            "keyword_precision",
            "keyword_recall",
            "keyword_f1",
            "llm_correctness",
            "llm_relevance",
            "llm_completeness",
            "llm_faithfulness",
            "llm_reason",
        }
        assert expected_keys.issubset(result.keys())

    def test_no_llm_judge_gives_none_scores(self):
        emb = self._embeddings([1.0, 0.0])
        result = compute_scores("machine learning", "machine learning", emb, emb)
        assert result["llm_correctness"] is None
        assert result["llm_relevance"] is None
        assert result["llm_completeness"] is None
        assert result["llm_faithfulness"] is None

    def test_llm_judge_ok_merges_scores(self):
        emb = self._embeddings([1.0, 0.0])
        judge = {
            "status": "ok",
            "correctness": 0.9,
            "relevance": 0.85,
            "completeness": 0.8,
            "faithfulness": 0.95,
            "reason": "Good answer.",
        }
        result = compute_scores("machine learning", "machine learning", emb, emb, judge)
        assert result["llm_correctness"] == 0.9
        assert result["llm_relevance"] == 0.85
        assert result["llm_reason"] == "Good answer."

    def test_llm_judge_failed_gives_none(self):
        emb = self._embeddings([1.0, 0.0])
        judge = {"status": "failed", "error": "API timeout"}
        result = compute_scores("machine learning", "machine learning", emb, emb, judge)
        assert result["llm_correctness"] is None
        assert result["llm_reason"] is None

    def test_no_arbitrary_composite_score(self):
        """There must be no 'final_score' key — it was removed as uncalibrated."""
        emb = self._embeddings([1.0, 0.0])
        result = compute_scores("text", "text", emb, emb)
        assert "final_score" not in result

    def test_no_hallucination_penalty_key(self):
        """'hallucination_penalty' was removed — it was keyword overlap misnamed."""
        emb = self._embeddings([1.0, 0.0])
        result = compute_scores("text", "text", emb, emb)
        assert "hallucination_penalty" not in result

    def test_empty_answer_does_not_crash(self):
        emb_empty = self._embeddings([0.0, 0.0])
        emb_ref = self._embeddings([1.0, 0.0])
        result = compute_scores("", "machine learning is great", emb_empty, emb_ref)
        assert isinstance(result, dict)
        assert result["keyword_precision"] is None  # no keywords in empty answer


# ---------------------------------------------------------------------------
# _avg aggregation helper
# ---------------------------------------------------------------------------
class TestAvgHelper:
    def test_basic(self):
        assert _avg([0.5, 0.5, 0.5]) == 0.5

    def test_ignores_none(self):
        assert _avg([0.5, None, 0.5]) == 0.5

    def test_all_none(self):
        assert _avg([None, None]) is None

    def test_empty_list(self):
        assert _avg([]) is None

    def test_single_value(self):
        assert _avg([0.75]) == 0.75

    def test_rounding(self):
        result = _avg([1/3, 1/3, 1/3])
        assert result == round(1/3, 4)


# ---------------------------------------------------------------------------
# _aggregate (aggregate metrics over CaseResults)
# ---------------------------------------------------------------------------
class TestAggregateMetrics:
    def _make_result(self, status="ok", sem=0.8, kw_f1=0.7, llm_c=0.9):
        return CaseResult(
            case_id="test",
            question="q",
            ground_truth="gt",
            status=status,
            semantic_similarity=sem if status == "ok" else None,
            keyword_f1=kw_f1 if status == "ok" else None,
            llm_correctness=llm_c if status == "ok" else None,
        )

    def test_empty_results(self):
        agg = _aggregate([])
        assert agg.count == 0
        assert agg.semantic_similarity is None

    def test_failed_cases_excluded(self):
        results = [
            self._make_result("ok"),
            self._make_result("failed"),
            self._make_result("ok"),
        ]
        agg = _aggregate(results)
        assert agg.count == 2

    def test_averages_correct(self):
        results = [
            self._make_result("ok", sem=0.6, kw_f1=0.4, llm_c=0.7),
            self._make_result("ok", sem=0.8, kw_f1=0.6, llm_c=0.9),
        ]
        agg = _aggregate(results)
        assert agg.semantic_similarity == 0.7
        assert agg.keyword_f1 == 0.5
        assert agg.llm_correctness == 0.8

    def test_all_failed_returns_zero_count(self):
        results = [
            self._make_result("failed"),
            self._make_result("failed"),
        ]
        agg = _aggregate(results)
        assert agg.count == 0
        assert agg.semantic_similarity is None
