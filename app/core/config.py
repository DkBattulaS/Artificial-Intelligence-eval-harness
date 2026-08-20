"""
Central configuration loaded from environment variables / .env file.

All application settings live here. No magic strings scattered through code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Settings:
    # --- LLM provider ---
    groq_api_key: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    groq_model: str = field(default_factory=lambda: os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"))

    # --- Embedding model ---
    embedding_model: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    )

    # --- Evaluation engine ---
    eval_concurrency: int = field(
        default_factory=lambda: int(os.getenv("EVAL_CONCURRENCY", "5"))
    )
    eval_timeout_seconds: int = field(
        default_factory=lambda: int(os.getenv("EVAL_TIMEOUT_SECONDS", "30"))
    )
    eval_max_retries: int = field(
        default_factory=lambda: int(os.getenv("EVAL_MAX_RETRIES", "2"))
    )
    run_llm_judge: bool = field(
        default_factory=lambda: os.getenv("RUN_LLM_JUDGE", "true").lower() == "true"
    )

    # --- Persistence ---
    db_path: str = field(
        default_factory=lambda: os.getenv("DB_PATH", "eval_runs.db")
    )

    # --- API ---
    cors_origins: list[str] = field(
        default_factory=lambda: [
            o.strip()
            for o in os.getenv("CORS_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000").split(",")
            if o.strip()
        ]
    )


# Singleton — import this everywhere instead of reading os.getenv directly
settings = Settings()
