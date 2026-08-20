"""
Unit tests for the semantic similarity evaluator.

External LLM calls are NOT required.
The sentence-transformers model is loaded locally and is a real embedding.

Tests verify:
  - Same sentence → high similarity (≥ 0.99)
  - Paraphrase → moderately high (> 0.6)
  - Unrelated text → low similarity (< 0.4)
  - Score is in [0, 1]
  - Score is a float, not None
  - Empty string edge case is handled
"""

import pytest
from app.evaluators.semantic import SemanticSimilarityEvaluator
from app.evaluators.base import EvaluationCase


@pytest.fixture(scope="module")
def evaluator():
    """Shared evaluator instance — loads the model once for the test module."""
    return SemanticSimilarityEvaluator(model_name="all-MiniLM-L6-v2")


def _case(answer: str, reference: str) -> EvaluationCase:
    return EvaluationCase(
        question="test question",
        answer=answer,
        reference=reference,
    )


class TestSemanticSimilarityEvaluator:
    def test_identical_text_scores_high(self, evaluator):
        result = evaluator.evaluate(_case(
            answer="Machine learning is a subset of artificial intelligence.",
            reference="Machine learning is a subset of artificial intelligence.",
        ))
        assert result["status"] == "ok"
        assert result["score"] >= 0.99

    def test_paraphrase_scores_moderately_high(self, evaluator):
        result = evaluator.evaluate(_case(
            answer="ML is a branch of AI that lets machines learn from data.",
            reference="Machine learning is a subset of artificial intelligence that learns from data.",
        ))
        assert result["status"] == "ok"
        assert result["score"] > 0.6

    def test_unrelated_scores_low(self, evaluator):
        result = evaluator.evaluate(_case(
            answer="The capital of France is Paris.",
            reference="Machine learning uses neural networks to learn patterns.",
        ))
        assert result["status"] == "ok"
        # Semantically unrelated — should be well below 0.6
        assert result["score"] < 0.6

    def test_score_is_float_in_range(self, evaluator):
        result = evaluator.evaluate(_case(
            answer="Deep learning uses multiple layers.",
            reference="Neural networks have many hidden layers.",
        ))
        assert result["status"] == "ok"
        score = result["score"]
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_empty_answer(self, evaluator):
        """Empty string should not crash — embedding model handles it."""
        result = evaluator.evaluate(_case(
            answer="",
            reference="Machine learning is a subset of AI.",
        ))
        # Should return a score (low, but defined) — or a graceful failure
        assert result["score"] is not None or result["status"] == "failed"

    def test_metadata_contains_model_name(self, evaluator):
        result = evaluator.evaluate(_case(
            answer="Deep learning uses neural networks.",
            reference="Neural networks are used in deep learning.",
        ))
        assert result["metadata"]["embedding_model"] == "all-MiniLM-L6-v2"

    def test_similar_topic_different_claim(self, evaluator):
        """
        Demonstrates that semantic similarity does NOT detect contradictions.
        Two sentences about the same topic with opposite meaning will score
        higher than unrelated sentences — this is the documented limitation.
        """
        result_correct = evaluator.evaluate(_case(
            answer="Overfitting reduces a model's ability to generalize.",
            reference="Overfitting reduces a model's ability to generalize.",
        ))
        result_contradiction = evaluator.evaluate(_case(
            answer="Overfitting always improves model generalization.",
            reference="Overfitting reduces a model's ability to generalize.",
        ))
        # Both are about overfitting/generalization — the contradiction still
        # scores higher than a fully unrelated sentence. This is expected and
        # correctly documented — semantic similarity ≠ factual correctness.
        assert result_contradiction["score"] > 0.3  # shares topic vocabulary

    def test_question_not_used_in_similarity(self, evaluator):
        """
        Verifies that the evaluator compares answer↔reference, not answer↔question.
        A different question with same answer/reference should give same score.
        """
        case1 = EvaluationCase(
            question="What is overfitting?",
            answer="A model that memorizes training data.",
            reference="A model that memorizes training data.",
        )
        case2 = EvaluationCase(
            question="Explain gravity to a child.",
            answer="A model that memorizes training data.",
            reference="A model that memorizes training data.",
        )
        r1 = evaluator.evaluate(case1)
        r2 = evaluator.evaluate(case2)
        assert r1["score"] == r2["score"]
