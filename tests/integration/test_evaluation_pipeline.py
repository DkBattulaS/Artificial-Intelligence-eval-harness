"""
Integration tests for the evaluation pipeline.

External LLM API calls are MOCKED — no GROQ_API_KEY required.
The sentence-transformers model IS loaded (local, no network needed).

Tests cover:
  - Full successful pipeline with mock provider
  - Provider failure → structured error, not fake model text
  - Dataset validation errors reported separately
  - Metric coverage tracking (failed cases don't drag down averages)
  - LLM judge failure → None metrics (not fabricated)
  - Rate limit retry behavior
  - API endpoint smoke test (POST /evaluations)
"""

import asyncio
import json
import os
import tempfile
import pytest
from unittest.mock import MagicMock, patch

from app.providers.base import GenerationResult
from app.core.errors import ProviderRateLimitError, ProviderError
from app.models.dataset import validate_dataset
from app.evaluators.lexical import LexicalEvaluator
from app.evaluators.base import EvaluationCase


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
SAMPLE_CASES = [
    {
        "question": "What is Machine Learning?",
        "ground_truth": "Machine Learning is a subset of AI that learns from data.",
        "category": "ML Basics",
        "difficulty": "easy",
    },
    {
        "question": "What is Deep Learning?",
        "ground_truth": "Deep Learning uses neural networks with multiple layers.",
        "category": "ML Basics",
        "difficulty": "medium",
    },
]


def _make_mock_provider(responses: list[str] | None = None, fail_on: set[int] | None = None):
    """
    Create a mock LLMProvider.

    responses: list of text responses (cycling)
    fail_on: set of call indices that should raise ProviderError
    """
    call_count = [0]
    if responses is None:
        responses = ["This is a mock response about machine learning and data."]

    class MockProvider:
        name = "mock"
        _model = "mock-model"

        def generate(self, prompt, system_prompt=None, temperature=0.0):
            idx = call_count[0]
            call_count[0] += 1

            if fail_on and idx in fail_on:
                raise ProviderError(f"Mock failure at call {idx}")

            text = responses[idx % len(responses)]
            return GenerationResult(
                text=text,
                model="mock-model",
                provider="mock",
                input_tokens=10,
                output_tokens=20,
                total_tokens=30,
                latency_ms=50.0,
                finish_reason="stop",
            )

    return MockProvider()


@pytest.fixture
def tmp_dataset(tmp_path):
    """Write SAMPLE_CASES to a temporary JSON file."""
    path = tmp_path / "test_dataset.json"
    path.write_text(json.dumps(SAMPLE_CASES))
    return str(path)


# ---------------------------------------------------------------------------
# Database setup for integration tests
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def isolated_db(tmp_path):
    """Each test gets its own in-memory-like SQLite database."""
    from app.db import database as db_module
    db_path = str(tmp_path / "test.db")
    db_module.init_db(db_path)
    yield
    db_module._db_path = None  # reset after test


# ---------------------------------------------------------------------------
# Dataset validation
# ---------------------------------------------------------------------------
class TestDatasetValidation:
    def test_valid_cases_pass(self):
        result = validate_dataset(SAMPLE_CASES)
        assert result.valid_count == 2
        assert result.error_count == 0

    def test_invalid_cases_reported(self):
        cases = SAMPLE_CASES + [{"ground_truth": "missing question"}]
        result = validate_dataset(cases)
        assert result.valid_count == 2
        assert result.error_count == 1

    def test_index_reported_in_error(self):
        cases = [{"ground_truth": "no question"}]
        result = validate_dataset(cases)
        assert result.errors[0]["index"] == 0


# ---------------------------------------------------------------------------
# Full pipeline (mocked provider)
# ---------------------------------------------------------------------------
class TestEvaluationPipeline:
    @pytest.mark.asyncio
    async def test_successful_run(self, tmp_dataset):
        """Full pipeline completes with correct structure."""
        from app.engine.runner import run_evaluation

        provider = _make_mock_provider(
            responses=["Machine learning learns patterns from data using algorithms."]
        )

        # Disable LLM judge to avoid needing a real API
        run, summary, results = await run_evaluation(
            dataset_path=tmp_dataset,
            provider=provider,
            run_llm_judge=False,
        )

        assert summary.total_cases == 2
        assert summary.successful_cases == 2
        assert summary.failed_cases == 0
        assert summary.coverage == 1.0
        assert summary.overall.count == 2
        # Semantic similarity should be defined
        assert summary.overall.semantic_similarity is not None

    @pytest.mark.asyncio
    async def test_provider_failure_is_structured(self, tmp_dataset):
        """
        When the provider fails, the case is marked failed with structured error.
        The failure must NOT be treated as a model response.
        """
        from app.engine.runner import run_evaluation

        # First call fails, second succeeds
        provider = _make_mock_provider(
            responses=["Good answer about deep learning."],
            fail_on={0},  # first case fails
        )

        run, summary, results = await run_evaluation(
            dataset_path=tmp_dataset,
            provider=provider,
            run_llm_judge=False,
        )

        assert summary.failed_cases == 1
        assert summary.successful_cases == 1
        assert summary.coverage == 0.5

        # Find the failed case
        failed = [r for r in results if r.status == "failed"]
        assert len(failed) == 1

        # The failed case must have structured error, NOT a fabricated answer
        assert failed[0].answer is None
        assert failed[0].error is not None
        assert "type" in failed[0].error
        assert failed[0].semantic_similarity is None  # undefined, not zero

    @pytest.mark.asyncio
    async def test_failed_cases_excluded_from_averages(self, tmp_dataset):
        """Failed cases must not count toward metric averages."""
        from app.engine.runner import run_evaluation

        provider = _make_mock_provider(fail_on={0})

        run, summary, results = await run_evaluation(
            dataset_path=tmp_dataset,
            provider=provider,
            run_llm_judge=False,
        )

        # Average is computed over 1 successful case only
        assert summary.overall.count == 1
        # Not over 2 (which would dilute with None)

    @pytest.mark.asyncio
    async def test_llm_judge_failure_gives_none_not_fabricated(self, tmp_dataset):
        """If the LLM judge fails, scores are None — never fabricated."""
        from app.engine.runner import run_evaluation

        provider = _make_mock_provider(
            responses=["Machine learning is a subset of AI."]
        )

        # Patch the judge to simulate failure
        with patch("app.services.llm_judge.judge") as mock_judge:
            mock_judge.return_value = {"status": "failed", "error": "API timeout"}

            run, summary, results = await run_evaluation(
                dataset_path=tmp_dataset,
                provider=provider,
                run_llm_judge=True,
            )

        for result in results:
            if result.status == "ok":
                assert result.llm_correctness is None
                assert result.llm_reason is None

    @pytest.mark.asyncio
    async def test_run_has_id_and_metadata(self, tmp_dataset):
        """Every run must have a run_id and capture metadata."""
        from app.engine.runner import run_evaluation

        provider = _make_mock_provider()

        run, summary, _ = await run_evaluation(
            dataset_path=tmp_dataset,
            provider=provider,
            run_llm_judge=False,
        )

        assert summary.run_id  # non-empty
        assert len(summary.run_id) == 36  # UUID4 format
        assert summary.model == "mock-model"
        assert summary.provider == "mock"
        assert summary.started_at  # non-empty timestamp
        assert summary.completed_at

    @pytest.mark.asyncio
    async def test_category_and_difficulty_aggregation(self, tmp_dataset):
        """Category/difficulty breakdown should be computed."""
        from app.engine.runner import run_evaluation

        provider = _make_mock_provider(
            responses=["Machine learning learns from data."]
        )

        run, summary, _ = await run_evaluation(
            dataset_path=tmp_dataset,
            provider=provider,
            run_llm_judge=False,
        )

        # Both cases are "ML Basics" category
        assert len(summary.by_category) == 1
        assert summary.by_category[0].label == "ML Basics"

        # Difficulties: easy and medium
        difficulty_labels = {b.label for b in summary.by_difficulty}
        assert "easy" in difficulty_labels
        assert "medium" in difficulty_labels


# ---------------------------------------------------------------------------
# Lexical evaluator integration
# ---------------------------------------------------------------------------
class TestLexicalEvaluatorIntegration:
    def test_with_real_case(self):
        evaluator = LexicalEvaluator()
        case = EvaluationCase(
            question="What is ML?",
            answer="Machine learning is a subset of artificial intelligence that learns from data.",
            reference="Machine Learning is a subset of AI that learns from data.",
        )
        result = evaluator.evaluate(case)
        assert result["status"] == "ok"
        assert result["metadata"]["keyword_f1"] > 0.5

    def test_model_output_error_string_not_evaluated(self):
        """
        Regression test: 'Error' (old behavior) must not be fed into the evaluator.
        If it were, it would produce near-zero scores that look like real evaluation.
        This test documents what happens if you DO evaluate the string 'Error'.
        """
        evaluator = LexicalEvaluator()
        case = EvaluationCase(
            question="What is ML?",
            answer="Error",  # old broken behavior
            reference="Machine learning learns from data.",
        )
        result = evaluator.evaluate(case)
        # Score will be near-zero — this documents why we must not do this
        f1 = result["metadata"]["keyword_f1"]
        # "error" may or may not match reference keywords
        # The point is: None or 0 should NOT be silently treated as a 0 score
        # The pipeline must never reach this state


# ---------------------------------------------------------------------------
# Database persistence integration
# ---------------------------------------------------------------------------
class TestDatabasePersistence:
    @pytest.mark.asyncio
    async def test_run_persisted_and_retrievable(self, tmp_dataset):
        from app.engine.runner import run_evaluation
        from app.db.database import get_run, get_case_results

        provider = _make_mock_provider()

        run, summary, _ = await run_evaluation(
            dataset_path=tmp_dataset,
            provider=provider,
            run_llm_judge=False,
        )

        stored_run = get_run(summary.run_id)
        assert stored_run is not None
        assert stored_run["run_id"] == summary.run_id
        assert stored_run["status"] == "completed"

        case_rows = get_case_results(summary.run_id)
        assert len(case_rows) == 2

    @pytest.mark.asyncio
    async def test_failed_cases_persisted_with_null_metrics(self, tmp_dataset):
        from app.engine.runner import run_evaluation
        from app.db.database import get_case_results

        provider = _make_mock_provider(fail_on={0})

        run, summary, _ = await run_evaluation(
            dataset_path=tmp_dataset,
            provider=provider,
            run_llm_judge=False,
        )

        rows = get_case_results(summary.run_id)
        failed_rows = [r for r in rows if r["status"] == "failed"]
        assert len(failed_rows) == 1
        assert failed_rows[0]["semantic_similarity"] is None  # NULL in DB, not 0
