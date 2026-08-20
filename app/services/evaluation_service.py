"""
Evaluation service: orchestrates per-case metric computation.

Metrics produced:
  semantic_similarity  – cosine similarity between answer and reference embeddings
                         using sentence-transformers. Measures meaning proximity,
                         NOT correctness.
  keyword_precision    – fraction of answer keywords found in the reference
  keyword_recall       – fraction of reference keywords found in the answer
  keyword_f1           – harmonic mean of keyword_precision and keyword_recall

These are the only metrics computed without an LLM call.
LLM-judge metrics (correctness, relevance, completeness, faithfulness) are
computed separately in llm_judge.py and merged here when available.

No composite "final_score" is produced because there is no calibrated weighting.
"""

from __future__ import annotations

import re
import logging
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer, util

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Stopwords (used only for keyword metrics, not semantic)
# ---------------------------------------------------------------------------
_STOPWORDS: frozenset[str] = frozenset({
    "is", "a", "the", "of", "and", "to", "in", "that", "it", "an", "are",
    "for", "on", "with", "as", "by", "this", "was", "be", "or", "at", "from",
    "but", "not", "have", "had", "has", "its", "we", "they", "you", "he",
    "she", "do", "did", "will", "would", "could", "should", "may", "might",
    "can", "into", "more", "also", "than", "when", "where", "which", "who",
    "so", "if", "up", "out", "about", "what", "all", "been",
})

# ---------------------------------------------------------------------------
# Embedding model (lazy-loaded on first use, configurable via env)
# ---------------------------------------------------------------------------
_model: Optional[SentenceTransformer] = None


def get_embedding_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    """Return a cached SentenceTransformer instance. Loaded once per process."""
    global _model
    if _model is None:
        logger.info("Loading embedding model: %s", model_name)
        _model = SentenceTransformer(model_name)
    return _model


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------
def normalize(text: str) -> str:
    """Lowercase, collapse whitespace."""
    return re.sub(r"\s+", " ", text.strip().lower())


def extract_keywords(text: str) -> frozenset[str]:
    """
    Extract non-stopword alphabetic tokens of length > 2.
    Returns an empty frozenset for empty/whitespace-only input.
    """
    words = re.findall(r"\b[a-z]+\b", normalize(text))
    return frozenset(w for w in words if w not in _STOPWORDS and len(w) > 2)


# ---------------------------------------------------------------------------
# Individual metrics
# ---------------------------------------------------------------------------
def compute_semantic_similarity(emb_answer, emb_reference) -> float:
    """
    Cosine similarity between two sentence embeddings.
    Range: [-1, 1] in theory; [0, 1] in practice for these models.
    Indicates *meaning proximity*, not factual correctness.
    """
    return round(float(util.cos_sim(emb_answer, emb_reference).item()), 4)


def compute_keyword_precision(answer_kw: frozenset[str], reference_kw: frozenset[str]) -> Optional[float]:
    """
    Of the keywords in the answer, what fraction appear in the reference?
    Returns None if the answer has no keywords (undefined, not zero).
    """
    if not answer_kw:
        return None
    return round(len(answer_kw & reference_kw) / len(answer_kw), 4)


def compute_keyword_recall(answer_kw: frozenset[str], reference_kw: frozenset[str]) -> Optional[float]:
    """
    Of the keywords in the reference, what fraction appear in the answer?
    Returns None if the reference has no keywords (undefined, not zero).
    """
    if not reference_kw:
        return None
    return round(len(answer_kw & reference_kw) / len(reference_kw), 4)


def compute_keyword_f1(precision: Optional[float], recall: Optional[float]) -> Optional[float]:
    """
    Harmonic mean of keyword precision and recall.
    Returns None if either component is None or their sum is zero.
    """
    if precision is None or recall is None:
        return None
    denom = precision + recall
    if denom == 0.0:
        return 0.0
    return round(2 * precision * recall / denom, 4)


# ---------------------------------------------------------------------------
# Per-case scoring
# ---------------------------------------------------------------------------
def compute_scores(
    answer: str,
    reference: str,
    emb_answer,
    emb_reference,
    llm_judge_result: Optional[dict] = None,
) -> dict:
    """
    Compute all non-LLM metrics and merge LLM judge results if provided.

    Parameters
    ----------
    answer        : normalized model answer
    reference     : normalized ground truth / reference
    emb_answer    : pre-computed answer embedding tensor
    emb_reference : pre-computed reference embedding tensor
    llm_judge_result : optional dict from llm_judge.judge(); keys:
                       correctness, relevance, completeness, faithfulness, reason

    Returns
    -------
    dict with all metric keys; any undefined metric is None (never fabricated).
    """
    answer_kw = extract_keywords(answer)
    reference_kw = extract_keywords(reference)

    semantic = compute_semantic_similarity(emb_answer, emb_reference)
    kw_precision = compute_keyword_precision(answer_kw, reference_kw)
    kw_recall = compute_keyword_recall(answer_kw, reference_kw)
    kw_f1 = compute_keyword_f1(kw_precision, kw_recall)

    result: dict = {
        # Semantic metric
        "semantic_similarity": semantic,
        # Lexical metrics
        "keyword_precision": kw_precision,
        "keyword_recall": kw_recall,
        "keyword_f1": kw_f1,
        # LLM-judge metrics (None if judge was not run or failed)
        "llm_correctness": None,
        "llm_relevance": None,
        "llm_completeness": None,
        "llm_faithfulness": None,
        "llm_reason": None,
    }

    if llm_judge_result and llm_judge_result.get("status") == "ok":
        result["llm_correctness"] = llm_judge_result.get("correctness")
        result["llm_relevance"] = llm_judge_result.get("relevance")
        result["llm_completeness"] = llm_judge_result.get("completeness")
        result["llm_faithfulness"] = llm_judge_result.get("faithfulness")
        result["llm_reason"] = llm_judge_result.get("reason")

    return result


# ---------------------------------------------------------------------------
# Batch evaluation pipeline
# ---------------------------------------------------------------------------
def evaluate_responses(data: list[dict], run_llm_judge: bool = True) -> list[dict]:
    """
    Evaluate a list of {question, answer, ground_truth} dicts.

    - Only items with status="ok" are scored; failed items carry their error.
    - Embeddings are computed in a single batch for efficiency.
    - LLM judge is called per-case only when run_llm_judge=True and the
      answer is available.

    Returns a list of result dicts (one per input item).
    """
    from app.services.llm_judge import judge  # local import avoids circular dep

    model = get_embedding_model()

    # Split into succeeded and failed cases
    ok_indices = [i for i, d in enumerate(data) if d.get("status") == "ok"]
    results: list[dict] = []

    if ok_indices:
        answers = [normalize(data[i]["answer"]) for i in ok_indices]
        references = [normalize(data[i]["ground_truth"]) for i in ok_indices]

        logger.info("Encoding %d answer/reference pairs", len(ok_indices))
        emb_answers = model.encode(answers, convert_to_tensor=True)
        emb_references = model.encode(references, convert_to_tensor=True)

    for i, item in enumerate(data):
        base = {
            "question": item["question"],
            "answer": item.get("answer"),
            "ground_truth": item.get("ground_truth"),
            "category": item.get("category"),
            "difficulty": item.get("difficulty"),
            "status": item.get("status", "ok"),
            "error": item.get("error"),
        }

        if item.get("status") != "ok":
            # Failed generation — metrics are undefined
            base.update({
                "semantic_similarity": None,
                "keyword_precision": None,
                "keyword_recall": None,
                "keyword_f1": None,
                "llm_correctness": None,
                "llm_relevance": None,
                "llm_completeness": None,
                "llm_faithfulness": None,
                "llm_reason": None,
            })
            results.append(base)
            continue

        # Find the batch index for this ok item
        batch_idx = ok_indices.index(i)
        ans_norm = answers[batch_idx]
        ref_norm = references[batch_idx]
        emb_a = emb_answers[batch_idx]
        emb_r = emb_references[batch_idx]

        llm_result = None
        if run_llm_judge:
            try:
                llm_result = judge(
                    question=item["question"],
                    answer=ans_norm,
                    reference=ref_norm,
                )
            except Exception as exc:
                logger.warning("LLM judge failed for question %r: %s", item["question"][:60], exc)

        scores = compute_scores(ans_norm, ref_norm, emb_a, emb_r, llm_result)
        base.update(scores)
        results.append(base)

    return results
