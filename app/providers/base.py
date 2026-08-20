"""
Abstract LLM provider interface.

All provider implementations must subclass LLMProvider and implement
`generate()`.  The evaluation engine depends only on this interface, not
on any concrete provider SDK.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class GenerationResult:
    """
    Structured output from a single LLM generation call.

    Fields
    ------
    text          : the generated text
    model         : model identifier used for this call
    provider      : provider name (e.g. "groq", "openai")
    input_tokens  : number of prompt tokens (None if provider doesn't report)
    output_tokens : number of completion tokens (None if provider doesn't report)
    total_tokens  : total tokens (None if provider doesn't report)
    latency_ms    : wall-clock call duration in milliseconds
    finish_reason : e.g. "stop", "length", "content_filter"
    """
    text: str
    model: str
    provider: str
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    latency_ms: Optional[float] = None
    finish_reason: Optional[str] = None


class LLMProvider(ABC):
    """
    Base class for all LLM provider adapters.

    Subclasses are responsible for:
    - Authenticating with the provider
    - Converting GenerationResult fields from the provider SDK response
    - Raising app-level exceptions (ProviderError subclasses) on failure
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable provider name, e.g. 'groq'."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
    ) -> GenerationResult:
        """
        Generate a completion for the given prompt.

        Parameters
        ----------
        prompt        : user message content
        system_prompt : optional system message
        temperature   : sampling temperature (0.0 = deterministic)

        Returns
        -------
        GenerationResult

        Raises
        ------
        ProviderAuthError      – bad/missing credentials
        ProviderRateLimitError – rate limit hit
        ProviderTimeoutError   – request timed out
        ProviderError          – any other provider-side failure
        """
