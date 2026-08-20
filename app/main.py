"""
AI Evaluation Harness — application entry point.

Start the server:
  uvicorn app.main:app --reload

Environment variables (see .env.example):
  GROQ_API_KEY       – required
  GROQ_MODEL         – default: llama-3.3-70b-versatile
  EMBEDDING_MODEL    – default: all-MiniLM-L6-v2
  EVAL_CONCURRENCY   – default: 5
  RUN_LLM_JUDGE      – default: true
  CORS_ORIGINS       – comma-separated list, default: localhost origins
  DB_PATH            – default: eval_runs.db
"""

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from app.core.logging import configure_logging
from app.core.config import settings
from app.db.database import init_db
from app.api.routes.evaluations import router as eval_router

# Configure logging before anything else
configure_logging()

app = FastAPI(
    title="AI Evaluation Harness",
    description=(
        "A trustworthy, extensible framework for evaluating LLM responses "
        "against reference datasets using semantic, lexical, and LLM-judge metrics."
    ),
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Accept"],
)

# Initialize SQLite database on startup
@app.on_event("startup")
def startup_event():
    init_db()


# Mount static files for the dashboard
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", include_in_schema=False)
def serve_dashboard():
    return FileResponse("static/index.html")


# Register API routes
app.include_router(eval_router)
