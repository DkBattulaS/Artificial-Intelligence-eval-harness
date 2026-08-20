"""
Abstract evaluator interface.

Every evaluator takes an EvaluationCase and a model answer, and returns
a MetricResult.  Evaluators are independently testable and not coupled to
FastAPI or any provider SDK.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class MetricStatus(str, Enum):
    OK = "ok"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class MetricResult:
    """
    Result of a single evaluator run on a single case.

    Fields
    ------
    metric     : evaluator/metric name (snake_case)
    score      : numeric score in [0, 1], or None if evaluation failed/skipped
    explanation: human-readable reason for the score
    status     : ok | failed | skipped
    metadata   : any extra structured data (e.g. matched keywords, raw judge output)
    """
    metric: str
    score: Optional[float]
    explanation: Optional[str] = None
    status: MetricStatus = MetricStatus.OK
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationCase:
    """
    A single test case passed to evaluators.

    question        : the question / input prompt
    answer          : the model-generated answer
    reference       : ground truth / expected answer
    context         : optional retrieved context (for RAG evaluation)
    expected_keywords: optional list of expected keywords (for lexical evaluators)
    category        : optional category label from the dataset
    difficulty      : optional difficulty label from the dataset
    """
    question: str
    answer: str
    reference: str
    context: Optional[str] = None
    expected_keywords: Optional[list[str]] = None
    category: Optional[str] = None
    difficulty: Optional[str] = None


class Evaluator(ABC):
    """
    Base class for all evaluators.

    Subclasses must implement `evaluate()`.  The method must never raise —
    instead return a MetricResult with status=FAILED and score=None.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Metric name, e.g. 'semantic_similarity'."""

    @abstractmethod
    def evaluate(self, case: EvaluationCase) -> MetricResult:
        """
        Evaluate the case and return a MetricResult.

        Must not raise exceptions — failures must be represented in the
        returned MetricResult with status=FAILED.
        """
