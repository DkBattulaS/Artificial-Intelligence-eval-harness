"""
Application-level exception hierarchy.

Raising specific exception types lets the API layer return the right
HTTP status codes without inspecting error message strings.
"""

from __future__ import annotations


class EvalHarnessError(Exception):
    """Base class for all application errors."""


class DatasetValidationError(EvalHarnessError):
    """Raised when a dataset fails schema validation."""


class ProviderError(EvalHarnessError):
    """Raised when an LLM provider call fails."""


class ProviderAuthError(ProviderError):
    """Missing or invalid API credentials."""


class ProviderRateLimitError(ProviderError):
    """Provider rate limit exceeded."""


class ProviderTimeoutError(ProviderError):
    """Provider call timed out."""


class EvaluatorError(EvalHarnessError):
    """Raised when an evaluator fails to produce a result."""


class RunNotFoundError(EvalHarnessError):
    """Requested evaluation run does not exist."""
