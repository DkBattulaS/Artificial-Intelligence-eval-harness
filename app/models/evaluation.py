"""
Evaluation result models.

CaseResult    – result for a single test case (one model response + all metrics)
RunSummary    – aggregate metrics and metadata for a full evaluation run
"""

from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field


class CaseResult(BaseModel):
    """
    Full result for a single test case.

    status: "ok" if the model responded successfully; "failed" otherwise.
    error:  structured error info when status="failed".
    Metric values are None when the case failed or the evaluator was skipped.
    """

    case_id: str
    question: str
    answer: Optional[str] = None
    ground_truth: str
    category: Optional[str] = None
    difficulty: Optional[str] = None

    # Generation metadata
    model: Optional[str] = None
    provider: Optional[str] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    latency_ms: Optional[float] = None
    finish_reason: Optional[str] = None

    # Execution status
    status: str = "ok"  # "ok" | "failed"
    error: Optional[dict] = None  # {"type": str, "message": str}

    # --- Metrics (None = undefined, not zero) ---
    # Semantic
    semantic_similarity: Optional[float] = None

    # Lexical
    keyword_precision: Optional[float] = None
    keyword_recall: Optional[float] = None
    keyword_f1: Optional[float] = None

    # LLM judge
    llm_correctness: Optional[float] = None
    llm_relevance: Optional[float] = None
    llm_completeness: Optional[float] = None
    llm_faithfulness: Optional[float] = None
    llm_reason: Optional[str] = None


class AggregateMetrics(BaseModel):
    """Averaged metrics over a set of CaseResults (successful cases only)."""
    count: int = 0
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


class CategoryBreakdown(BaseModel):
    """Per-category or per-difficulty aggregate metrics."""
    label: str
    metrics: AggregateMetrics


class RunSummary(BaseModel):
    """
    Summary of a complete evaluation run.
    Returned by GET /evaluations/{run_id}.
    """
    run_id: str
    dataset: str
    model: str
    provider: str
    embedding_model: str
    llm_judge_enabled: bool
    temperature: float = 0.0
    started_at: str
    completed_at: Optional[str] = None
    duration_seconds: Optional[float] = None

    total_cases: int
    successful_cases: int
    failed_cases: int
    coverage: float  # successful / total

    overall: AggregateMetrics
    by_category: list[CategoryBreakdown] = Field(default_factory=list)
    by_difficulty: list[CategoryBreakdown] = Field(default_factory=list)

    dataset_validation_errors: int = 0
