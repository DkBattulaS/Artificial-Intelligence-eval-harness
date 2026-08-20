"""
SQLite persistence layer using Python's stdlib sqlite3.

No ORM. Keeps the project lightweight and easy to run locally.

Schema
------
evaluation_runs  – one row per run (metadata + status)
case_results     – one row per test case result (FK to evaluation_runs)

The database file path is configured via settings.db_path (default: eval_runs.db).
The schema is created on first use via init_db().
"""

from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from typing import Generator, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
_SCHEMA = """
CREATE TABLE IF NOT EXISTS evaluation_runs (
    run_id                    TEXT PRIMARY KEY,
    status                    TEXT NOT NULL,
    dataset                   TEXT NOT NULL,
    model                     TEXT NOT NULL,
    provider                  TEXT NOT NULL,
    embedding_model           TEXT NOT NULL,
    llm_judge_enabled         INTEGER NOT NULL DEFAULT 1,
    started_at                TEXT NOT NULL,
    completed_at              TEXT,
    total_cases               INTEGER NOT NULL DEFAULT 0,
    successful_cases          INTEGER NOT NULL DEFAULT 0,
    failed_cases              INTEGER NOT NULL DEFAULT 0,
    dataset_validation_errors INTEGER NOT NULL DEFAULT 0,
    error_message             TEXT
);

CREATE TABLE IF NOT EXISTS case_results (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id           TEXT NOT NULL REFERENCES evaluation_runs(run_id),
    case_id          TEXT NOT NULL,
    question         TEXT NOT NULL,
    answer           TEXT,
    ground_truth     TEXT NOT NULL,
    category         TEXT,
    difficulty       TEXT,
    model            TEXT,
    provider         TEXT,
    input_tokens     INTEGER,
    output_tokens    INTEGER,
    total_tokens     INTEGER,
    latency_ms       REAL,
    finish_reason    TEXT,
    status           TEXT NOT NULL DEFAULT 'ok',
    error            TEXT,
    -- metric columns (NULL = undefined, not zero)
    semantic_similarity  REAL,
    keyword_precision    REAL,
    keyword_recall       REAL,
    keyword_f1           REAL,
    llm_correctness      REAL,
    llm_relevance        REAL,
    llm_completeness     REAL,
    llm_faithfulness     REAL,
    llm_reason           TEXT
);

CREATE INDEX IF NOT EXISTS idx_case_results_run_id ON case_results(run_id);
CREATE INDEX IF NOT EXISTS idx_case_results_category ON case_results(category);
CREATE INDEX IF NOT EXISTS idx_case_results_difficulty ON case_results(difficulty);
"""

# ---------------------------------------------------------------------------
# Connection management
# ---------------------------------------------------------------------------
_db_path: Optional[str] = None


def init_db(db_path: Optional[str] = None) -> None:
    """
    Initialize the database. Creates tables if they don't exist.
    Call once at application startup.
    """
    global _db_path
    if db_path is None:
        from app.core.config import settings
        db_path = settings.db_path
    _db_path = db_path
    logger.info("Initializing database at %s", _db_path)
    with _get_connection() as conn:
        conn.executescript(_SCHEMA)


def _get_connection() -> sqlite3.Connection:
    if _db_path is None:
        raise RuntimeError("Database not initialized. Call init_db() first.")
    conn = sqlite3.connect(_db_path, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")  # safe for concurrent reads
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


@contextmanager
def get_db() -> Generator[sqlite3.Connection, None, None]:
    """Context manager that yields a connection and commits on exit (or rolls back on error)."""
    conn = _get_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Run CRUD
# ---------------------------------------------------------------------------
def insert_run(run: dict) -> None:
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO evaluation_runs
              (run_id, status, dataset, model, provider, embedding_model,
               llm_judge_enabled, started_at, completed_at,
               total_cases, successful_cases, failed_cases,
               dataset_validation_errors, error_message)
            VALUES
              (:run_id, :status, :dataset, :model, :provider, :embedding_model,
               :llm_judge_enabled, :started_at, :completed_at,
               :total_cases, :successful_cases, :failed_cases,
               :dataset_validation_errors, :error_message)
            """,
            run,
        )


def update_run(run_id: str, fields: dict) -> None:
    """Update specific fields on an existing run row."""
    if not fields:
        return
    set_clause = ", ".join(f"{k} = :{k}" for k in fields)
    fields["run_id"] = run_id
    with get_db() as conn:
        conn.execute(
            f"UPDATE evaluation_runs SET {set_clause} WHERE run_id = :run_id",
            fields,
        )


def get_run(run_id: str) -> Optional[dict]:
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM evaluation_runs WHERE run_id = ?", (run_id,)
        ).fetchone()
    return dict(row) if row else None


def list_runs(limit: int = 50, offset: int = 0) -> list[dict]:
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM evaluation_runs ORDER BY started_at DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Case result CRUD
# ---------------------------------------------------------------------------
def insert_case_result(run_id: str, result: dict) -> None:
    """Insert a single case result. error dict is serialized to JSON string."""
    row = {**result, "run_id": run_id}
    if isinstance(row.get("error"), dict):
        row["error"] = json.dumps(row["error"])
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO case_results
              (run_id, case_id, question, answer, ground_truth,
               category, difficulty, model, provider,
               input_tokens, output_tokens, total_tokens,
               latency_ms, finish_reason, status, error,
               semantic_similarity, keyword_precision, keyword_recall, keyword_f1,
               llm_correctness, llm_relevance, llm_completeness, llm_faithfulness,
               llm_reason)
            VALUES
              (:run_id, :case_id, :question, :answer, :ground_truth,
               :category, :difficulty, :model, :provider,
               :input_tokens, :output_tokens, :total_tokens,
               :latency_ms, :finish_reason, :status, :error,
               :semantic_similarity, :keyword_precision, :keyword_recall, :keyword_f1,
               :llm_correctness, :llm_relevance, :llm_completeness, :llm_faithfulness,
               :llm_reason)
            """,
            row,
        )


def insert_case_results_batch(run_id: str, results: list[dict]) -> None:
    """Insert multiple case results in a single transaction."""
    with get_db() as conn:
        for result in results:
            row = {**result, "run_id": run_id}
            if isinstance(row.get("error"), dict):
                row["error"] = json.dumps(row["error"])
            conn.execute(
                """
                INSERT INTO case_results
                  (run_id, case_id, question, answer, ground_truth,
                   category, difficulty, model, provider,
                   input_tokens, output_tokens, total_tokens,
                   latency_ms, finish_reason, status, error,
                   semantic_similarity, keyword_precision, keyword_recall, keyword_f1,
                   llm_correctness, llm_relevance, llm_completeness, llm_faithfulness,
                   llm_reason)
                VALUES
                  (:run_id, :case_id, :question, :answer, :ground_truth,
                   :category, :difficulty, :model, :provider,
                   :input_tokens, :output_tokens, :total_tokens,
                   :latency_ms, :finish_reason, :status, :error,
                   :semantic_similarity, :keyword_precision, :keyword_recall, :keyword_f1,
                   :llm_correctness, :llm_relevance, :llm_completeness, :llm_faithfulness,
                   :llm_reason)
                """,
                row,
            )


def get_case_results(run_id: str) -> list[dict]:
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM case_results WHERE run_id = ? ORDER BY id",
            (run_id,),
        ).fetchall()
    results = []
    for row in rows:
        r = dict(row)
        if r.get("error") and isinstance(r["error"], str):
            try:
                r["error"] = json.loads(r["error"])
            except (json.JSONDecodeError, TypeError):
                pass
        results.append(r)
    return results
