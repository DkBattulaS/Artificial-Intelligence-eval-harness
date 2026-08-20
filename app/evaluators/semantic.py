"""
Semantic similarity evaluator.

Computes cosine similarity between sentence embeddings of the answer and
the reference using sentence-transformers.

What this measures:
  Meaning proximity between two texts.

What this does NOT measure:
  - Factual correctness
  - Whether the answer addresses the question
  - Whether claims are grounded in source material

The embedding model is configurable. The model name used is recorded in
metadata so results are reproducible across model changes.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class SemanticSimilarityEvaluator:
    """
    Evaluates semantic similarity between answer and reference.

    Parameters
    ----------
    model_name : sentence-transformers model identifier.
                 Defaults to settings.embedding_model.
    """

    name = "semantic_similarity"

    def __init__(self, model_name: Optional[str] = None) -> None:
        from app.core.config import settings
        self._model_name = model_name or settings.embedding_model
        self._model = None  # lazy load

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading embedding model: %s", self._model_name)
            self._model = SentenceTransformer(self._model_name)
        return self._model

    def evaluate(self, case) -> dict:
        """
        Returns a dict compatible with MetricResult fields.
        score: cosine similarity in [0, 1]
        """
        from sentence_transformers import util
        try:
            model = self._get_model()
            emb = model.encode([case.answer, case.reference], convert_to_tensor=True)
            score = round(float(util.cos_sim(emb[0], emb[1]).item()), 4)
            return {
                "metric": self.name,
                "score": score,
                "explanation": f"Cosine similarity between answer and reference embeddings ({self._model_name}).",
                "status": "ok",
                "metadata": {"embedding_model": self._model_name},
            }
        except Exception as exc:
            logger.error("SemanticSimilarityEvaluator failed: %s", exc)
            return {
                "metric": self.name,
                "score": None,
                "explanation": str(exc),
                "status": "failed",
                "metadata": {},
            }

    def encode_batch(self, texts: list[str]):
        """
        Batch-encode a list of texts. Used by the engine for efficiency.
        Returns a tensor of embeddings.
        """
        return self._get_model().encode(texts, convert_to_tensor=True)
