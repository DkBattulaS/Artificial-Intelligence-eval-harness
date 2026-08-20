"""
Groq LLM provider implementation.

Uses the official `groq` Python SDK.
Translates Groq-specific exceptions into app-level ProviderError subclasses
so the engine doesn't need to know about the SDK.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from app.providers.base import GenerationResult, LLMProvider
from app.core.errors import ProviderAuthError, ProviderError, ProviderRateLimitError, ProviderTimeoutError

logger = logging.getLogger(__name__)


class GroqProvider(LLMProvider):
    """
    Adapter for the Groq API (https://console.groq.com).

    Parameters
    ----------
    api_key : Groq API key. If None, reads GROQ_API_KEY from environment.
    model   : Model identifier. Defaults to llama-3.3-70b-versatile.
    """

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None) -> None:
        import os
        from groq import Groq

        _api_key = api_key or os.getenv("GROQ_API_KEY", "")
        if not _api_key:
            raise ProviderAuthError(
                "GROQ_API_KEY is not set. "
                "Add it to your .env file or export it in your shell."
            )
        self._client = Groq(api_key=_api_key)
        self._model = model or os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    @property
    def name(self) -> str:
        return "groq"

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
    ) -> GenerationResult:
        """
        Call the Groq chat completions API.

        Maps Groq SDK exceptions to ProviderError subclasses:
          - AuthenticationError  → ProviderAuthError
          - RateLimitError       → ProviderRateLimitError
          - APITimeoutError      → ProviderTimeoutError
          - everything else      → ProviderError
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        t0 = time.monotonic()
        try:
            response = self._client.chat.completions.create(
                messages=messages,
                model=self._model,
                temperature=temperature,
            )
        except Exception as exc:
            latency_ms = round((time.monotonic() - t0) * 1000, 1)
            self._raise_mapped(exc, latency_ms)

        latency_ms = round((time.monotonic() - t0) * 1000, 1)

        choice = response.choices[0]
        usage = response.usage

        return GenerationResult(
            text=choice.message.content.strip(),
            model=response.model,
            provider=self.name,
            input_tokens=usage.prompt_tokens if usage else None,
            output_tokens=usage.completion_tokens if usage else None,
            total_tokens=usage.total_tokens if usage else None,
            latency_ms=latency_ms,
            finish_reason=choice.finish_reason,
        )

    @staticmethod
    def _raise_mapped(exc: Exception, latency_ms: float) -> None:
        """Re-raise the Groq exception as an app-level ProviderError."""
        exc_type = type(exc).__name__
        msg = str(exc)

        logger.debug("Groq API error (%.0fms): %s: %s", latency_ms, exc_type, msg)

        if "AuthenticationError" in exc_type or "401" in msg:
            raise ProviderAuthError(f"Groq authentication failed: {msg}") from exc
        if "RateLimitError" in exc_type or "429" in msg:
            raise ProviderRateLimitError(f"Groq rate limit exceeded: {msg}") from exc
        if "Timeout" in exc_type or "timeout" in msg.lower():
            raise ProviderTimeoutError(f"Groq request timed out: {msg}") from exc
        raise ProviderError(f"Groq API error ({exc_type}): {msg}") from exc
