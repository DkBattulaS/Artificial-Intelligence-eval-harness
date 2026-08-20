"""
Pydantic schemas for the evaluations API.

Request and response models are separate from the internal domain models
so the API contract is explicit and versioning is straightforward.
"""

from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------
class CreateEvaluationRequest(BaseModel):
    """
    POST /evaluations request body.

    dataset : path or name of the dataset to evaluate.
              Defaults to the built-in sample testset.
    model   : model identifier to use. Defaults to GROQ_MODEL env var.
    run_llm_judge : whether to run the LLM judge. Defaults to RUN_LLM_JUDGE env var.
    """
    dataset: str = Field(
        default="tests/sample_testset.json",
        description="Relative path to a JSON dataset file.",
    )
    model: Optional[str] = Field(
        default=None,
        description="Override the LLM model. Defaults to GROQ_MODEL env var.",
    )
    run_llm_judge: Optional[bool] = Field(
        default=None,
        description="Whether to run the LLM-as-a-judge evaluator.",
    )


# ---------------------------------------------------------------------------
# Responses
# ---------------------------------------------------------------------------
class AggregateMetricsResponse(BaseModel):
    count: int
    semantic_similarity: Optional[float] = None
    keyword_precision: Optional[float] = None
    keyword_recall: Optional[float] = None
    keyword_f1: Optional[float] = None
    llm_correctness: Optional[float] = None
    llm_relevance: Optional[float] = None
    llm_completeness: Optional[float] = None
    llm_faithfulness: Optional[float] = None
    avg_latency_ms: Optional[float] = None
    total_tokens: Optional[int] = None


class CategoryBreakdownResponse(BaseModel):
    label: str
    metrics: AggregateMetricsResponse


class EvaluationRunResponse(BaseModel):
    """Returned by POST /evaluations and GET /evaluations/{run_id}."""
    run_id: str
    status: str
    dataset: str
    model: str
    provider: str
    embedding_model: str
    llm_judge_enabled: bool
    started_at: str
    completed_at: Optional[str] = None
    duration_seconds: Optional[float] = None
    total_cases: int
    successful_cases: int
    failed_cases: int
    coverage: float
    dataset_validation_errors: int
    overall: Optional[AggregateMetricsResponse] = None
    by_category: list[CategoryBreakdownResponse] = Field(default_factory=list)
    by_difficulty: list[CategoryBreakdownResponse] = Field(default_factory=list)


class CaseResultResponse(BaseModel):
    """Returned by GET /evaluations/{run_id}/results (one item per case)."""
    case_id: str
    question: str
    answer: Optional[str] = None
    ground_truth: str
    category: Optional[str] = None
    difficulty: Optional[str] = None
    model: Optional[str] = None
    provider: Optional[str] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    latency_ms: Optional[float] = None
    finish_reason: Optional[str] = None
    status: str
    error: Optional[Any] = None
    semantic_similarity: Optional[float] = None
    keyword_precision: Optional[float] = None
    keyword_recall: Optional[float] = None
    keyword_f1: Optional[float] = None
    llm_correctness: Optional[float] = None
    llm_relevance: Optional[float] = None
    llm_completeness: Optional[float] = None
    llm_faithfulness: Optional[float] = None
    llm_reason: Optional[str] = None


class EvaluationResultsResponse(BaseModel):
    """Returned by GET /evaluations/{run_id}/results."""
    run_id: str
    total: int
    results: list[CaseResultResponse]


class RunListResponse(BaseModel):
    """Returned by GET /evaluations."""
    runs: list[EvaluationRunResponse]
    total: int
