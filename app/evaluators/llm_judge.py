"""
LLM-as-a-Judge evaluator.

Wraps the judge service (app.services.llm_judge) into the Evaluator interface.
This evaluator makes a real LLM API call.

Scores returned:
  llm_correctness  : factual correctness vs. reference
  llm_relevance    : does the answer address the question?
  llm_completeness : coverage of key points from the reference
  llm_faithfulness : answer stays within what the reference supports

All scores are in [0, 1] as returned by the judge LLM.
If the judge fails (API error, malformed output), all scores are None
and status is "failed" — scores are never fabricated.

Limitations:
  - Scores depend on the quality of the judge model.
  - Reference-bias: answers that differ from the reference in wording
    but are equally correct may score lower.
  - The judge model is the same as the generation model by default
    (self-evaluation bias). This is a known limitation and is documented
    in run metadata.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class LLMJudgeEvaluator:
    """
    Evaluates answer quality using a real LLM API call.

    Parameters
    ----------
    provider : LLMProvider instance. If None, uses the default GroqProvider.
    """

    name = "llm_judge"

    def __init__(self, provider=None) -> None:
        self._provider = provider  # injected or lazily created

    def _get_provider(self):
        if self._provider is None:
            from app.providers.groq import GroqProvider
            self._provider = GroqProvider()
        return self._provider

    def evaluate(self, case) -> dict:
        """
        Calls the LLM judge service and returns structured scores.
        Never raises — failures are represented in the returned dict.
        """
        from app.services.llm_judge import judge
        try:
            result = judge(
                question=case.question,
                answer=case.answer,
                reference=case.reference,
                context=case.context,
            )
        except Exception as exc:
            logger.error("LLMJudgeEvaluator unexpected error: %s", exc)
            result = {"status": "failed", "error": str(exc)}

        if result.get("status") == "ok":
            return {
                "metric": self.name,
                "score": result.get("llm_correctness"),  # primary score
                "explanation": result.get("reason"),
                "status": "ok",
                "metadata": {
                    "llm_correctness": result.get("correctness"),
                    "llm_relevance": result.get("relevance"),
                    "llm_completeness": result.get("completeness"),
                    "llm_faithfulness": result.get("faithfulness"),
                    "reason": result.get("reason"),
                },
            }
        else:
            return {
                "metric": self.name,
                "score": None,
                "explanation": result.get("error"),
                "status": "failed",
                "metadata": {},
            }
