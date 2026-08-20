"""
EvaluationRun tracks the lifecycle of a single evaluation execution.

States:
  pending   – created, not yet started
  running   – currently executing
  completed – all cases processed (some may have failed)
  failed    – run itself failed (e.g. dataset could not be loaded)
"""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel


class EvaluationRun(BaseModel):
    run_id: str
    status: str  # pending | running | completed | failed
    dataset: str
    model: str
    provider: str
    embedding_model: str
    llm_judge_enabled: bool
    started_at: str
    completed_at: Optional[str] = None
    total_cases: int = 0
    successful_cases: int = 0
    failed_cases: int = 0
    dataset_validation_errors: int = 0
    error_message: Optional[str] = None
