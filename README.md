# AI Evaluation Harness

A trustworthy, extensible framework for evaluating LLM responses against reference datasets using semantic similarity, lexical overlap, and a real LLM-as-a-judge.

---

## What it is

A FastAPI application that:

1. Loads a JSON test dataset (question + ground truth pairs)
2. Sends each question to an LLM provider (currently Groq)
3. Evaluates each response against the reference using three independent evaluators
4. Persists results to a local SQLite database
5. Exposes a REST API and a minimal web dashboard

This project is designed to be readable, honest about its metric limitations, and easy to extend or contribute to.

---

## Why it exists

Most LLM evaluation tooling is either too opinionated (large framework lock-in), too complex to run locally (heavy infrastructure), or uses metrics that sound sophisticated but are poorly defined.

This harness prioritizes:
- Correct evaluation methodology over impressive-sounding metrics
- Reproducible runs with captured metadata
- Honest documentation of what each metric measures and does not measure
- A codebase that an engineer can read, understand, and extend in an afternoon

---

## Architecture

```
app/
├── main.py                   Entry point (FastAPI app, startup hooks)
├── core/
│   ├── config.py             Central settings (read from env)
│   ├── logging.py            Logging configuration
│   └── errors.py             Application exception hierarchy
├── providers/
│   ├── base.py               LLMProvider abstract interface + GenerationResult
│   └── groq.py               Groq provider implementation
├── evaluators/
│   ├── base.py               Evaluator interface + EvaluationCase + MetricResult
│   ├── semantic.py           Semantic similarity (sentence-transformers)
│   ├── lexical.py            Keyword precision / recall / F1
│   └── llm_judge.py          LLM-as-a-judge evaluator
├── engine/
│   └── runner.py             Async evaluation orchestrator
├── models/
│   ├── dataset.py            TestCase Pydantic model + dataset validator
│   ├── evaluation.py         CaseResult, RunSummary, AggregateMetrics
│   └── run.py                EvaluationRun lifecycle model
├── api/
│   ├── routes/evaluations.py REST API routes
│   └── schemas/evaluation.py Request/response Pydantic schemas
├── db/
│   └── database.py           SQLite persistence (stdlib sqlite3)
└── services/
    ├── evaluation_service.py Metric computation helpers
    ├── llm_judge.py          Judge prompt + output parsing
    └── llm_service.py        Raw Groq client (legacy, thin wrapper)

static/
└── index.html                Dashboard UI

tests/
├── unit/
│   ├── test_lexical.py
│   ├── test_semantic.py
│   ├── test_dataset.py
│   ├── test_scoring.py
│   └── test_normalization.py
├── integration/
│   └── test_evaluation_pipeline.py
└── fixtures/
    └── sample_dataset.json
```

---

## Supported providers

| Provider | Status | Notes |
|----------|--------|-------|
| Groq     | ✅ Implemented | `llama-3.3-70b-versatile` default |
| OpenAI   | 🔲 Planned | Provider interface is ready |
| Ollama   | 🔲 Planned | Provider interface is ready |

Adding a new provider requires implementing `LLMProvider` from `app/providers/base.py`.

---

## Supported evaluators

| Evaluator | Metric(s) | What it measures | What it does NOT measure |
|-----------|-----------|------------------|--------------------------|
| Semantic similarity | `semantic_similarity` | Cosine similarity between answer and reference embeddings | Factual correctness, question relevance |
| Lexical | `keyword_precision`, `keyword_recall`, `keyword_f1` | Surface-form token overlap | Meaning, correctness, hallucination |
| LLM judge | `llm_correctness`, `llm_relevance`, `llm_completeness`, `llm_faithfulness` | Judge model's structured assessment | Objective ground truth (model-dependent) |

There is no composite "final score". Independent metrics are exposed separately. Combining them requires calibration with labeled data, which is out of scope for this project.

---

## Dataset format

```json
[
  {
    "id": "optional_stable_id",
    "question": "What is Machine Learning?",
    "ground_truth": "Machine Learning is a subset of AI that learns from data.",
    "context": "optional retrieved context for RAG evaluation",
    "expected_keywords": ["learning", "subset", "data"],
    "category": "ML Basics",
    "difficulty": "easy"
  }
]
```

Required fields: `question`, `ground_truth`

Optional fields: `id`, `context`, `expected_keywords`, `category`, `difficulty`

`difficulty` must be one of: `easy`, `medium`, `hard`

Invalid cases are reported before the run starts. The run proceeds with valid cases only.

---

## Installation

```bash
git clone https://github.com/DkBattulaS/Artificial-Intelligence-eval-harness.git
cd Artificial-Intelligence-eval-harness

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

---

## Configuration

```bash
cp .env.example .env
```

Edit `.env` and set at minimum:

```
GROQ_API_KEY=your_groq_api_key_here
```

See `.env.example` for all configuration options with documentation.

---

## Running an evaluation

Start the server:

```bash
uvicorn app.main:app --reload
```

Then open [http://localhost:8000](http://localhost:8000) in your browser and click **Run evaluation**.

Or use the API directly:

```bash
curl -s -X POST http://localhost:8000/evaluations \
     -H "Content-Type: application/json" \
     -d '{}' | python -m json.tool
```

To disable the LLM judge (faster, no extra API calls):

```bash
curl -s -X POST http://localhost:8000/evaluations \
     -H "Content-Type: application/json" \
     -d '{"run_llm_judge": false}' | python -m json.tool
```

---

## API

### `POST /evaluations`

Start an evaluation run. Returns the completed run summary.

Request body (all optional):
```json
{
  "dataset": "tests/sample_testset.json",
  "model": "llama-3.3-70b-versatile",
  "run_llm_judge": true
}
```

Response: `EvaluationRunResponse` (includes overall metrics, by-category, by-difficulty breakdowns)

### `GET /evaluations`

List recent evaluation runs.

Query params: `limit` (default 20), `offset` (default 0)

### `GET /evaluations/{run_id}`

Get the summary and aggregate metrics for a specific run.

### `GET /evaluations/{run_id}/results`

Get per-case results for a specific run.

Interactive API docs: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## Metrics and their actual meaning

### `semantic_similarity`

Cosine similarity between sentence-transformer embeddings of the answer and the reference.

- Range: [0, 1]
- High score means the texts are semantically close in embedding space
- Does NOT imply factual correctness
- A fluent wrong answer about the same topic can score high
- A correct paraphrase with different vocabulary may score lower than expected
- Model: configurable via `EMBEDDING_MODEL` (default: `all-MiniLM-L6-v2`)

### `keyword_precision`

Fraction of keywords in the answer that appear in the reference.

- Range: [0, 1] or `null` if the answer has no keywords
- High precision means the answer doesn't introduce many terms not in the reference
- Does NOT detect hallucinations — a factually wrong answer using reference vocabulary scores high

### `keyword_recall`

Fraction of keywords in the reference that appear in the answer.

- Range: [0, 1] or `null` if the reference has no keywords
- High recall means the answer covers the key terms of the reference

### `keyword_f1`

Harmonic mean of keyword precision and recall.

- Range: [0, 1]
- A balanced measure of lexical coverage
- Only meaningful as a lexical overlap metric, not semantic correctness

### `llm_correctness`, `llm_relevance`, `llm_completeness`, `llm_faithfulness`

Scores from a real LLM judge (Groq/Llama) that evaluates the answer against the reference using a structured prompt.

- Range: [0, 1] each
- `null` if the judge was disabled or the judge call failed
- Scores depend on the judge model's quality and consistency
- Known limitation: using the same model for generation and judging introduces self-evaluation bias
- The judge uses the reference as ground truth — correct answers that differ in wording may score lower

---

## Limitations

1. **Single provider**: Only Groq is currently implemented. The provider interface is extensible.

2. **Reference-only evaluation**: All evaluators compare answer against the reference answer. There is no retrieval quality evaluation for RAG pipelines (the infrastructure is in place; `EvaluationCase.context` is supported by the LLM judge).

3. **Self-evaluation bias**: The same model used for generation is used for judging by default. This is a known limitation of LLM-as-a-judge approaches.

4. **No calibration**: The metrics have not been calibrated against human judgments. Absolute scores should be treated as directional indicators rather than absolute quality measurements.

5. **Lexical metrics are surface-form only**: Keyword F1 measures token overlap, not meaning. A correct paraphrase and a wrong answer with the right vocabulary can score identically.

6. **Synchronous API**: The `POST /evaluations` endpoint runs the full evaluation synchronously. For large datasets this may take several minutes. A job queue would be appropriate for production use.

---

## Development

```bash
# Run tests (no API key required — LLM calls are mocked)
pytest

# Run with verbose output
pytest -v

# Run only unit tests
pytest tests/unit/

# Run only integration tests
pytest tests/integration/
```

---

## Testing

Tests are in `tests/unit/` and `tests/integration/`.

External LLM calls are mocked in all tests — no `GROQ_API_KEY` required to run the test suite.

The sentence-transformers embedding model is loaded locally (downloads on first run, ~90MB).

---

## Future roadmap

- [ ] OpenAI and Ollama provider implementations
- [ ] RAG retrieval quality evaluators (context precision, context recall)
- [ ] Async background evaluation with status polling
- [ ] Model comparison: evaluate multiple models on the same dataset in one run
- [ ] Claim extraction for factuality evaluation
- [ ] Export results to CSV / JSON
- [ ] Configurable metric weights with documented calibration methodology

---

## Contributing

Contributions are welcome. Please ensure:

- New metrics are honestly described (what they measure and what they don't)
- Tests cover normal cases, edge cases, and failure modes
- No new heavy dependencies without justification
- The project remains runnable locally without external infrastructure

---

## License

MIT
