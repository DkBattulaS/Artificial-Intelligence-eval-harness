"""
RAG evaluation service — placeholder.

RAG-specific evaluation (retrieval quality, context faithfulness) is planned
for a future iteration. The evaluator abstraction in app/evaluators/ is
designed to support context-aware evaluation when implemented.

See: app/evaluators/base.py (EvaluationCase.context field)
     app/services/llm_judge.py (context parameter in judge())
"""
