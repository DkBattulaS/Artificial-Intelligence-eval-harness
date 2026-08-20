"""
Evaluations API routes.

POST /evaluations              – start a new evaluation run
GET  /evaluations              – list recent runs
GET  /evaluations/{run_id}     – get run summary/metadata
GET  /evaluations/{run_id}/results – get per-case results
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from fastapi.responses import JSONResponse

from app.api.schemas.evaluation import (
    CreateEvaluationRequest,
    EvaluationRunResponse,
    EvaluationResultsResponse,
    CaseResultResponse,
    RunListResponse,
    AggregateMetricsResponse,
    CategoryBreakdownResponse,
)
from app.core.errors import RunNotFoundError

router = APIRouter(prefix="/evaluations", tags=["evaluations"])
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _run_to_response(run_dict: dict, summary=None) -> EvaluationRunResponse:
    """Convert a DB run dict (+ optional RunSummary) to response schema."""
    overall = None
    by_category = []
    by_difficulty = []

    if summary:
        overall = AggregateMetricsResponse(**summary.overall.model_dump())
        by_category = [
            CategoryBreakdownResponse(
                label=b.label,
                metrics=AggregateMetricsResponse(**b.metrics.model_dump()),
            )
            for b in summary.by_category
        ]
        by_difficulty = [
            CategoryBreakdownResponse(
                label=b.label,
                metrics=AggregateMetricsResponse(**b.metrics.model_dump()),
            )
            for b in summary.by_difficulty
        ]

    return EvaluationRunResponse(
        run_id=run_dict["run_id"],
        status=run_dict["status"],
        dataset=run_dict["dataset"],
        model=run_dict["model"],
        provider=run_dict["provider"],
        embedding_model=run_dict["embedding_model"],
        llm_judge_enabled=bool(run_dict["llm_judge_enabled"]),
        started_at=run_dict["started_at"],
        completed_at=run_dict.get("completed_at"),
        duration_seconds=None,
        total_cases=run_dict.get("total_cases", 0),
        successful_cases=run_dict.get("successful_cases", 0),
        failed_cases=run_dict.get("failed_cases", 0),
        coverage=(
            round(run_dict["successful_cases"] / run_dict["total_cases"], 4)
            if run_dict.get("total_cases", 0) > 0 else 0.0
        ),
        dataset_validation_errors=run_dict.get("dataset_validation_errors", 0),
        overall=overall,
        by_category=by_category,
        by_difficulty=by_difficulty,
    )


def _resolve_dataset_path(dataset: str) -> str:
    """Resolve a dataset path relative to the project root."""
    if os.path.isabs(dataset):
        return dataset
    # Try relative to CWD first, then relative to project root
    cwd_path = os.path.join(os.getcwd(), dataset)
    if os.path.exists(cwd_path):
        return cwd_path
    # Fallback: assume it's relative to the project root
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    root_path = os.path.join(root, dataset)
    if os.path.exists(root_path):
        return root_path
    raise FileNotFoundError(f"Dataset not found: {dataset!r}")


# ---------------------------------------------------------------------------
# POST /evaluations  — start a run
# ---------------------------------------------------------------------------
@router.post("", response_model=EvaluationRunResponse, status_code=202)
async def create_evaluation(body: CreateEvaluationRequest) -> EvaluationRunResponse:
    """
    Start a new evaluation run.

    Runs synchronously (within the async event loop with a semaphore) and
    returns the full summary when complete. For very large datasets this
    could take minutes — the 202 status signals that work is in progress
    even though we return the completed result in this response.

    A background job queue is not introduced to keep the project lightweight.
    """
    from app.engine.runner import run_evaluation
    from app.providers.groq import GroqProvider
    from app.core.config import settings
    from app.core.errors import ProviderAuthError

    # Resolve dataset path
    try:
        dataset_path = _resolve_dataset_path(body.dataset)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    # Build provider
    try:
        provider = GroqProvider(model=body.model or settings.groq_model)
    except ProviderAuthError as exc:
        raise HTTPException(status_code=401, detail=str(exc))

    run_llm_judge = body.run_llm_judge
    if run_llm_judge is None:
        run_llm_judge = settings.run_llm_judge

    try:
        run, summary, _cases = await run_evaluation(
            dataset_path=dataset_path,
            provider=provider,
            run_llm_judge=run_llm_judge,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except ProviderAuthError as exc:
        raise HTTPException(status_code=401, detail=str(exc))
    except Exception as exc:
        logger.error("Evaluation run failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {exc}")

    return EvaluationRunResponse(
        run_id=summary.run_id,
        status="completed",
        dataset=summary.dataset,
        model=summary.model,
        provider=summary.provider,
        embedding_model=summary.embedding_model,
        llm_judge_enabled=summary.llm_judge_enabled,
        started_at=summary.started_at,
        completed_at=summary.completed_at,
        duration_seconds=summary.duration_seconds,
        total_cases=summary.total_cases,
        successful_cases=summary.successful_cases,
        failed_cases=summary.failed_cases,
        coverage=summary.coverage,
        dataset_validation_errors=summary.dataset_validation_errors,
        overall=AggregateMetricsResponse(**summary.overall.model_dump()),
        by_category=[
            CategoryBreakdownResponse(
                label=b.label,
                metrics=AggregateMetricsResponse(**b.metrics.model_dump()),
            )
            for b in summary.by_category
        ],
        by_difficulty=[
            CategoryBreakdownResponse(
                label=b.label,
                metrics=AggregateMetricsResponse(**b.metrics.model_dump()),
            )
            for b in summary.by_difficulty
        ],
    )


# ---------------------------------------------------------------------------
# GET /evaluations  — list runs
# ---------------------------------------------------------------------------
@router.get("", response_model=RunListResponse)
def list_evaluations(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
) -> RunListResponse:
    from app.db.database import list_runs
    runs = list_runs(limit=limit, offset=offset)
    return RunListResponse(
        runs=[_run_to_response(r) for r in runs],
        total=len(runs),
    )


# ---------------------------------------------------------------------------
# GET /evaluations/{run_id}  — run summary
# ---------------------------------------------------------------------------
@router.get("/{run_id}", response_model=EvaluationRunResponse)
def get_evaluation(run_id: str) -> EvaluationRunResponse:
    from app.db.database import get_run, get_case_results
    from app.models.evaluation import CaseResult, AggregateMetrics, CategoryBreakdown
    from app.engine.runner import _aggregate, _breakdown_by

    run = get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Run {run_id!r} not found.")

    # Recompute aggregates from stored results (single source of truth)
    raw_results = get_case_results(run_id)
    case_results = [CaseResult(**r) for r in raw_results]

    overall = _aggregate(case_results)
    by_category = _breakdown_by(case_results, "category")
    by_difficulty = _breakdown_by(case_results, "difficulty")

    total = run.get("total_cases", 0)
    successful = run.get("successful_cases", 0)

    return EvaluationRunResponse(
        run_id=run["run_id"],
        status=run["status"],
        dataset=run["dataset"],
        model=run["model"],
        provider=run["provider"],
        embedding_model=run["embedding_model"],
        llm_judge_enabled=bool(run["llm_judge_enabled"]),
        started_at=run["started_at"],
        completed_at=run.get("completed_at"),
        duration_seconds=None,
        total_cases=total,
        successful_cases=successful,
        failed_cases=run.get("failed_cases", 0),
        coverage=round(successful / total, 4) if total > 0 else 0.0,
        dataset_validation_errors=run.get("dataset_validation_errors", 0),
        overall=AggregateMetricsResponse(**overall.model_dump()),
        by_category=[
            CategoryBreakdownResponse(
                label=b.label,
                metrics=AggregateMetricsResponse(**b.metrics.model_dump()),
            )
            for b in by_category
        ],
        by_difficulty=[
            CategoryBreakdownResponse(
                label=b.label,
                metrics=AggregateMetricsResponse(**b.metrics.model_dump()),
            )
            for b in by_difficulty
        ],
    )


# ---------------------------------------------------------------------------
# GET /evaluations/{run_id}/results  — per-case results
# ---------------------------------------------------------------------------
@router.get("/{run_id}/results", response_model=EvaluationResultsResponse)
def get_evaluation_results(run_id: str) -> EvaluationResultsResponse:
    from app.db.database import get_run, get_case_results

    run = get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Run {run_id!r} not found.")

    raw = get_case_results(run_id)
    results = [CaseResultResponse(**r) for r in raw]

    return EvaluationResultsResponse(
        run_id=run_id,
        total=len(results),
        results=results,
    )
