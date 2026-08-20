"""
Dataset model with Pydantic validation.

A TestCase is the atomic unit of evaluation. Invalid test cases are
rejected with clear error messages before the evaluation pipeline runs.
The dataset loader validates every item and reports all failures upfront.
"""

from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field, field_validator, model_validator


class TestCase(BaseModel):
    """
    A single evaluation test case.

    Required fields:
      question     : the input prompt sent to the model
      ground_truth : the reference answer for evaluation

    Optional fields:
      id                : stable identifier (auto-generated from index if absent)
      context           : retrieved context for RAG evaluation
      expected_keywords : keywords the answer should contain
      category          : test case category (e.g. "ML Basics")
      difficulty        : "easy" | "medium" | "hard"
      metadata          : any extra data passed through unchanged
    """

    id: Optional[str] = None
    question: str = Field(..., min_length=1)
    ground_truth: str = Field(..., min_length=1)
    context: Optional[str] = None
    expected_keywords: Optional[list[str]] = None
    category: Optional[str] = None
    difficulty: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("question", "ground_truth", mode="before")
    @classmethod
    def strip_whitespace(cls, v: Any) -> Any:
        if isinstance(v, str):
            return v.strip()
        return v

    @field_validator("difficulty", mode="before")
    @classmethod
    def validate_difficulty(cls, v: Any) -> Any:
        if v is None:
            return v
        normalized = str(v).strip().lower()
        allowed = {"easy", "medium", "hard"}
        if normalized not in allowed:
            raise ValueError(f"difficulty must be one of {allowed}, got {v!r}")
        return normalized

    @field_validator("expected_keywords", mode="before")
    @classmethod
    def validate_keywords(cls, v: Any) -> Any:
        if v is None:
            return v
        if not isinstance(v, list):
            raise ValueError("expected_keywords must be a list of strings")
        return [str(k).strip() for k in v if str(k).strip()]

    model_config = {"extra": "allow"}  # tolerate unknown fields — store in metadata


class DatasetValidationResult(BaseModel):
    """Result of validating a dataset."""
    valid: list[TestCase]
    errors: list[dict]  # {"index": int, "error": str}
    total: int
    valid_count: int
    error_count: int


def validate_dataset(raw: list[dict]) -> DatasetValidationResult:
    """
    Validate a list of raw dicts as TestCase objects.

    Does NOT raise on individual item failures — it collects all errors
    and returns a DatasetValidationResult so the caller can decide whether
    to proceed with valid cases or abort the run.

    Parameters
    ----------
    raw : list of raw dicts from a JSON dataset file

    Returns
    -------
    DatasetValidationResult with .valid and .errors populated
    """
    from pydantic import ValidationError

    valid: list[TestCase] = []
    errors: list[dict] = []

    for i, item in enumerate(raw):
        # Assign id from index if not present
        if isinstance(item, dict) and "id" not in item:
            item = {**item, "id": f"case_{i:04d}"}

        try:
            case = TestCase.model_validate(item)
            valid.append(case)
        except ValidationError as exc:
            errors.append({
                "index": i,
                "error": str(exc),
                "raw": item,
            })
        except Exception as exc:
            errors.append({
                "index": i,
                "error": f"Unexpected error: {exc}",
                "raw": item,
            })

    return DatasetValidationResult(
        valid=valid,
        errors=errors,
        total=len(raw),
        valid_count=len(valid),
        error_count=len(errors),
    )
