"""
LLM provider service (Groq).

Exposes:
  get_llm_response(question)      – simple question → answer string for evaluation
  get_llm_response_raw(...)       – full message interface for the LLM judge

The Groq client is instantiated lazily on first call so that a missing
GROQ_API_KEY does not crash the server at startup — it surfaces as a
proper HTTP error at request time instead.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

_client = None


def _get_client():
    """Lazily initialize the Groq client. Raises RuntimeError if key is absent."""
    global _client
    if _client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. "
                "Create a .env file with GROQ_API_KEY=<your_key> or export it in your environment."
            )
        from groq import Groq  # import deferred so missing dep gives a clear message
        _client = Groq(api_key=api_key)
    return _client


def _default_model() -> str:
    return os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")


def get_llm_response(question: str) -> str:
    """
    Send a question to the configured LLM and return the text answer.

    Raises on failure — callers in the evaluation pipeline are expected to
    catch this and record a structured failure rather than using the
    exception message as an answer.
    """
    client = _get_client()
    model = _default_model()
    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": question}],
        model=model,
        temperature=0.0,  # deterministic for reproducibility
    )
    return chat_completion.choices[0].message.content.strip()


def get_llm_response_raw(
    system_prompt: str,
    user_content: str,
    model: Optional[str] = None,
) -> str:
    """
    Low-level chat completion with explicit system + user messages.
    Used by the LLM judge.  Returns the raw response text.

    Raises on API failure — the judge wraps this in a try/except.
    """
    client = _get_client()
    model = model or _default_model()
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        model=model,
        temperature=0.0,
    )
    return chat_completion.choices[0].message.content.strip()
