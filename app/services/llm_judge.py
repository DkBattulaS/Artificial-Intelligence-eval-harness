"""
LLM-as-a-Judge evaluator.

Uses the configured LLM provider to score a model answer against a reference
on four criteria:

  correctness  – is the answer factually correct relative to the reference?
  relevance    – does the answer address the question asked?
  completeness – does the answer cover the key points of the reference?
  faithfulness – is the answer grounded in the reference without adding
                 unsupported claims?

Each score is in [0.0, 1.0].  The judge returns structured JSON which is
parsed and validated here.  If the LLM returns malformed output, the call
fails explicitly rather than fabricating scores.

Limitations (documented honestly):
  - Scores depend on the quality and consistency of the judge model.
  - The judge uses the *reference* as the ground truth proxy, which means
    a correct answer that differs from the reference may score lower than
    it deserves (reference-bias).
  - For RAG evaluation with retrieved context, pass context via the
    `context` parameter instead of / in addition to `reference`.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------
_JUDGE_SYSTEM_PROMPT = """\
You are a rigorous, impartial evaluation judge for AI-generated answers.
You will be given a question, a reference answer, and a model answer.
Score the model answer on four criteria, each on a scale of 0.0 to 1.0:

  correctness  : Is the model answer factually correct relative to the reference?
  relevance    : Does the model answer actually address the question?
  completeness : Does the model answer cover the key points present in the reference?
  faithfulness : Does the model answer stick to what is supported by the reference,
                 without introducing unsupported claims?

Return ONLY a JSON object with exactly these keys:
  correctness, relevance, completeness, faithfulness, reason

- All score values must be numbers between 0.0 and 1.0 (inclusive).
- "reason" must be a concise 1-2 sentence explanation of the scores.
- Do not include any text outside the JSON object.
- Do not use markdown code fences.

Example output:
{"correctness": 0.9, "relevance": 1.0, "completeness": 0.8, "faithfulness": 0.95, "reason": "The answer correctly identifies the concept but omits one key detail from the reference."}
"""

_JUDGE_USER_TEMPLATE = """\
Question: {question}

Reference answer: {reference}

Model answer: {answer}
"""

# ---------------------------------------------------------------------------
# Required keys and valid score range
# ---------------------------------------------------------------------------
_REQUIRED_KEYS = {"correctness", "relevance", "completeness", "faithfulness", "reason"}
_SCORE_KEYS = {"correctness", "relevance", "completeness", "faithfulness"}


def _parse_and_validate(raw: str) -> dict:
    """
    Parse the judge LLM output. Returns validated dict or raises ValueError.

    Attempts to extract a JSON object even if the model wraps it in prose
    or code fences — but is strict about the schema once extracted.
    """
    # Strip optional markdown code fences
    clean = raw.strip()
    fence_match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", clean)
    if fence_match:
        clean = fence_match.group(1).strip()

    # Try to extract a JSON object if surrounded by prose
    obj_match = re.search(r"\{[\s\S]+\}", clean)
    if obj_match:
        clean = obj_match.group(0)

    try:
        parsed = json.loads(clean)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Judge returned non-JSON output: {exc}\nRaw: {raw[:300]}") from exc

    if not isinstance(parsed, dict):
        raise ValueError(f"Judge output is not a JSON object. Got: {type(parsed)}")

    missing = _REQUIRED_KEYS - parsed.keys()
    if missing:
        raise ValueError(f"Judge output missing keys: {missing}. Got: {list(parsed.keys())}")

    for key in _SCORE_KEYS:
        val = parsed[key]
        if not isinstance(val, (int, float)):
            raise ValueError(f"Judge score '{key}' must be a number, got {type(val)}: {val!r}")
        if not (0.0 <= float(val) <= 1.0):
            raise ValueError(f"Judge score '{key}' out of range [0,1]: {val}")
        parsed[key] = round(float(val), 4)

    if not isinstance(parsed.get("reason"), str):
        raise ValueError(f"Judge 'reason' must be a string, got: {type(parsed.get('reason'))}")

    return parsed


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def judge(
    question: str,
    answer: str,
    reference: str,
    context: Optional[str] = None,
) -> dict:
    """
    Call the LLM judge and return a validated result dict.

    Returns
    -------
    On success:
        {
            "status": "ok",
            "correctness": float,
            "relevance": float,
            "completeness": float,
            "faithfulness": float,
            "reason": str,
        }

    On failure:
        {
            "status": "failed",
            "error": str,
        }

    Never raises — caller decides how to handle failures.
    """
    from app.services.llm_service import get_llm_response_raw  # local import

    user_content = _JUDGE_USER_TEMPLATE.format(
        question=question,
        reference=reference if not context else f"{reference}\n\nContext: {context}",
        answer=answer,
    )

    try:
        raw = get_llm_response_raw(
            system_prompt=_JUDGE_SYSTEM_PROMPT,
            user_content=user_content,
        )
    except Exception as exc:
        logger.warning("LLM judge API call failed: %s", exc)
        return {"status": "failed", "error": str(exc)}

    try:
        parsed = _parse_and_validate(raw)
    except ValueError as exc:
        logger.warning("LLM judge output validation failed: %s", exc)
        return {"status": "failed", "error": str(exc)}

    return {"status": "ok", **parsed}
