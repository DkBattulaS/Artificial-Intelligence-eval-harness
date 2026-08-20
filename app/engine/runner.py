"""
Evaluation engine runner.

Orchestrates the full evaluation pipeline for a single run:

  1. Validate dataset
  2. For each test case:
     a. Call LLM provider → GenerationResult or structured failure
     b. Run evaluators (semantic, lexical, LLM judge)
  3. Aggregate metrics by overall / category / difficulty
  4. Persist results to database
  5. Return RunSummary

Design decisions:
  - Concurrency is controlled (settings.eval_concurrency max simultaneous LLM calls)
  - LLM calls have retry with exponential backoff for rate-limit errors
  - Evaluator failures produce None metrics, not fabricated scores
  - Generation failures are represented as structured error dicts, not as
    fake model text fed into evaluators
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Optional

from app.core.config import settings
from app.core.errors import ProviderRateLimitError, ProviderError
from app.models.dataset import validate_dataset, TestCase
from app.models.evaluation import (
    CaseResult, RunSummary, AggregateMetrics, CategoryBreakdown,
)
from app.models.run import EvaluationRun

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Retry helper
# ---------------------------------------------------------------------------
async def _call_with_retry(fn, *args, max_retries: int = 2, base_delay: float = 1.0):
    """
    Call fn(*args) with exponential backoff on ProviderRateLimitError.
    Other ProviderErrors are re-raised immediately.
    """
    for attempt in range(max_retries + 1):
        try:
            return await asyncio.get_event_loop().run_in_executor(None, fn, *args)
        except ProviderRateLimitError as exc:
            if attempt == max_retries:
                raise
            delay = base_delay * (2 ** attempt)
            logger.warning(
                "Rate limit hit (attempt %d/%d). Retrying in %.1fs: %s",
                attempt + 1, max_retries + 1, delay, exc,
            )
            await asyncio.sleep(delay)


# ---------------------------------------------------------------------------
# Single case execution
# ---------------------------------------------------------------------------
async def _run_case(
    case: TestCase,
    provider,
    semantic_evaluator,
    lexical_evaluator,
    llm_judge_evaluator,
    run_llm_judge: bool,
    semaphore: asyncio.Semaphore,
) -> CaseResult:
    """Execute one test case: generate + evaluate. Never raises."""

    async with semaphore:
        # --- Generation ---
        gen_result = None
        error_info = None

        try:
            gen_result = await _call_with_retry(
                provider.generate,
                case.question,
                None,   # no system prompt for evaluation generation
                0.0,    # temperature=0 for reproducibility
                max_retries=settings.eval_max_retries,
            )
        except ProviderError as exc:
            error_info = {
                "type": type(exc).__name__,
                "message": str(exc),
            }
            logger.warning("Generation failed for case %s: %s", case.id, exc)
        except Exception as exc:
            error_info = {
                "type": "UnexpectedError",
                "message": str(exc),
            }
            logger.error("Unexpected error for case %s: %s", case.id, exc, exc_info=True)

        if error_info or gen_result is None:
            return CaseResult(
                case_id=case.id or "",
                question=case.question,
                answer=None,
                ground_truth=case.ground_truth,
                category=case.category,
                difficulty=case.difficulty,
                status="failed",
                error=error_info,
            )

        # --- Build evaluator case object ---
        from app.evaluators.base import EvaluationCase as EvCase
        ev_case = EvCase(
            question=case.question,
            answer=gen_result.text,
            reference=case.ground_truth,
            context=case.context,
            expected_keywords=case.expected_keywords,
            category=case.category,
            difficulty=case.difficulty,
        )

        # --- Semantic similarity ---
        sem_result = await asyncio.get_event_loop().run_in_executor(
            None, semantic_evaluator.evaluate, ev_case
        )

        # --- Lexical metrics ---
        lex_result = await asyncio.get_event_loop().run_in_executor(
            None, lexical_evaluator.evaluate, ev_case
        )

        # --- LLM judge (optional) ---
        judge_result = None
        if run_llm_judge:
            try:
                judge_result = await _call_with_retry(
                    llm_judge_evaluator.evaluate,
                    ev_case,
                    max_retries=settings.eval_max_retries,
                )
            except Exception as exc:
                logger.warning("LLM judge failed for case %s: %s", case.id, exc)

        # --- Assemble CaseResult ---
        lex_meta = lex_result.get("metadata", {})

        return CaseResult(
            case_id=case.id or "",
            question=case.question,
            answer=gen_result.text,
            ground_truth=case.ground_truth,
            category=case.category,
            difficulty=case.difficulty,
            model=gen_result.model,
            provider=gen_result.provider,
            input_tokens=gen_result.input_tokens,
            output_tokens=gen_result.output_tokens,
            total_tokens=gen_result.total_tokens,
            latency_ms=gen_result.latency_ms,
            finish_reason=gen_result.finish_reason,
            status="ok",
            error=None,
            semantic_similarity=sem_result.get("score"),
            keyword_precision=lex_meta.get("keyword_precision"),
            keyword_recall=lex_meta.get("keyword_recall"),
            keyword_f1=lex_meta.get("keyword_f1"),
            llm_correctness=judge_result["metadata"].get("llm_correctness") if judge_result and judge_result.get("status") == "ok" else None,
            llm_relevance=judge_result["metadata"].get("llm_relevance") if judge_result and judge_result.get("status") == "ok" else None,
            llm_completeness=judge_result["metadata"].get("llm_completeness") if judge_result and judge_result.get("status") == "ok" else None,
            llm_faithfulness=judge_result["metadata"].get("llm_faithfulness") if judge_result and judge_result.get("status") == "ok" else None,
            llm_reason=judge_result["metadata"].get("reason") if judge_result and judge_result.get("status") == "ok" else None,
        )


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def _avg(values: list[Optional[float]]) -> Optional[float]:
    """Average of non-None values. Returns None if list is empty or all None."""
    non_null = [v for v in values if v is not None]
    if not non_null:
        return None
    return round(sum(non_null) / len(non_null), 4)


def _aggregate(results: list[CaseResult]) -> AggregateMetrics:
    """Aggregate metrics over a list of CaseResults (successful cases only)."""
    ok = [r for r in results if r.status == "ok"]
    if not ok:
        return AggregateMetrics(count=0)

    latencies = [r.latency_ms for r in ok if r.latency_ms is not None]
    tokens = [r.total_tokens for r in ok if r.total_tokens is not None]

    return AggregateMetrics(
        count=len(ok),
        semantic_similarity=_avg([r.semantic_similarity for r in ok]),
        keyword_precision=_avg([r.keyword_precision for r in ok]),
        keyword_recall=_avg([r.keyword_recall for r in ok]),
        keyword_f1=_avg([r.keyword_f1 for r in ok]),
        llm_correctness=_avg([r.llm_correctness for r in ok]),
        llm_relevance=_avg([r.llm_relevance for r in ok]),
        llm_completeness=_avg([r.llm_completeness for r in ok]),
        llm_faithfulness=_avg([r.llm_faithfulness for r in ok]),
        avg_latency_ms=_avg(latencies) if latencies else None,
        total_tokens=sum(tokens) if tokens else None,
    )


def _breakdown_by(results: list[CaseResult], key: str) -> list[CategoryBreakdown]:
    """Group results by category or difficulty and aggregate each group."""
    from collections import defaultdict
    groups: dict[str, list[CaseResult]] = defaultdict(list)
    for r in results:
        label = getattr(r, key) or "unknown"
        groups[label].append(r)
    return [
        CategoryBreakdown(label=label, metrics=_aggregate(items))
        for label, items in sorted(groups.items())
    ]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
async def run_evaluation(
    dataset_path: str,
    provider=None,
    run_llm_judge: Optional[bool] = None,
) -> tuple[EvaluationRun, RunSummary, list[CaseResult]]:
    """
    Execute a full evaluation run.

    Parameters
    ----------
    dataset_path  : path to the JSON dataset file
    provider      : LLMProvider instance (defaults to GroqProvider)
    run_llm_judge : override settings.run_llm_judge

    Returns
    -------
    (EvaluationRun, RunSummary, list[CaseResult])

    Persists results to the database.
    """
    import json
    from app.db.database import insert_run, update_run, insert_case_results_batch
    from app.evaluators.semantic import SemanticSimilarityEvaluator
    from app.evaluators.lexical import LexicalEvaluator
    from app.evaluators.llm_judge import LLMJudgeEvaluator

    if provider is None:
        from app.providers.groq import GroqProvider
        provider = GroqProvider()

    if run_llm_judge is None:
        run_llm_judge = settings.run_llm_judge

    run_id = str(uuid.uuid4())
    started_at = datetime.now(timezone.utc).isoformat()
    dataset_name = dataset_path.split("/")[-1].split("\\")[-1]

    # --- Create run record ---
    run = EvaluationRun(
        run_id=run_id,
        status="running",
        dataset=dataset_name,
        model=getattr(provider, "_model", settings.groq_model),
        provider=provider.name,
        embedding_model=settings.embedding_model,
        llm_judge_enabled=run_llm_judge,
        started_at=started_at,
    )
    insert_run(run.model_dump())
    logger.info("Started evaluation run %s", run_id)

    # --- Load and validate dataset ---
    try:
        with open(dataset_path, encoding="utf-8") as f:
            raw = json.load(f)
    except Exception as exc:
        update_run(run_id, {
            "status": "failed",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "error_message": f"Failed to load dataset: {exc}",
        })
        raise

    validation = validate_dataset(raw)
    if validation.errors:
        logger.warning(
            "Dataset validation: %d valid, %d invalid",
            validation.valid_count, validation.error_count,
        )
        for e in validation.errors:
            logger.warning("  Case %d invalid: %s", e["index"], e["error"])

    if not validation.valid:
        update_run(run_id, {
            "status": "failed",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "error_message": "No valid test cases in dataset.",
            "dataset_validation_errors": validation.error_count,
        })
        raise ValueError("Dataset has no valid test cases.")

    update_run(run_id, {
        "total_cases": validation.valid_count,
        "dataset_validation_errors": validation.error_count,
    })

    # --- Instantiate evaluators ---
    semantic_evaluator = SemanticSimilarityEvaluator(settings.embedding_model)
    lexical_evaluator = LexicalEvaluator()
    llm_judge_evaluator = LLMJudgeEvaluator(provider=provider)

    semaphore = asyncio.Semaphore(settings.eval_concurrency)

    # --- Run all cases concurrently (up to concurrency limit) ---
    t_start = time.monotonic()
    tasks = [
        _run_case(
            case=case,
            provider=provider,
            semantic_evaluator=semantic_evaluator,
            lexical_evaluator=lexical_evaluator,
            llm_judge_evaluator=llm_judge_evaluator,
            run_llm_judge=run_llm_judge,
            semaphore=semaphore,
        )
        for case in validation.valid
    ]
    case_results: list[CaseResult] = await asyncio.gather(*tasks)
    duration = round(time.monotonic() - t_start, 2)

    completed_at = datetime.now(timezone.utc).isoformat()
    successful = sum(1 for r in case_results if r.status == "ok")
    failed = sum(1 for r in case_results if r.status != "ok")
    coverage = round(successful / len(case_results), 4) if case_results else 0.0

    # --- Aggregate ---
    overall = _aggregate(case_results)
    by_category = _breakdown_by(case_results, "category")
    by_difficulty = _breakdown_by(case_results, "difficulty")

    summary = RunSummary(
        run_id=run_id,
        dataset=dataset_name,
        model=run.model,
        provider=run.provider,
        embedding_model=settings.embedding_model,
        llm_judge_enabled=run_llm_judge,
        started_at=started_at,
        completed_at=completed_at,
        duration_seconds=duration,
        total_cases=len(case_results),
        successful_cases=successful,
        failed_cases=failed,
        coverage=coverage,
        overall=overall,
        by_category=by_category,
        by_difficulty=by_difficulty,
        dataset_validation_errors=validation.error_count,
    )

    # --- Persist results ---
    insert_case_results_batch(run_id, [r.model_dump() for r in case_results])
    update_run(run_id, {
        "status": "completed",
        "completed_at": completed_at,
        "total_cases": len(case_results),
        "successful_cases": successful,
        "failed_cases": failed,
        "dataset_validation_errors": validation.error_count,
    })

    logger.info(
        "Run %s completed in %.1fs — %d/%d cases succeeded (coverage: %.0f%%)",
        run_id, duration, successful, len(case_results), coverage * 100,
    )

    return run, summary, case_results
