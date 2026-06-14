# Mental Health Support API

FastAPI backend for the Mental Health Support Assistant.  
Built with **Clean Architecture** — domain logic is fully decoupled from ML infrastructure and the web framework.

---

## Architecture

```
mental_health_api/
│
├── domain/                        # Enterprise rules — zero external deps
│   ├── entities/
│   │   └── suggestion.py          # Suggestion, SourceCard, QueryResult dataclasses
│   └── interfaces/
│       ├── retriever.py           # Port: EmbeddingRetriever (ABC)
│       ├── reranker.py            # Port: Reranker (ABC)
│       ├── llm_processor.py       # Port: LLMPostProcessor (ABC)
│       └── repository.py          # Port: SuggestionRepository (ABC)
│
├── application/                   # Application rules — depends only on domain
│   └── use_cases/
│       ├── query_support.py       # 6-stage RAG pipeline use case
│       └── get_tags.py            # List available tag filters
│
├── infrastructure/                # Concrete adapters — implements domain ports
│   ├── ml/
│   │   ├── faiss_retriever.py     # SentenceTransformer + FAISS
│   │   ├── cross_encoder_reranker.py  # ms-marco CrossEncoder
│   │   └── qwen_processor.py     # Qwen2.5-1.5B-Instruct (+ fallback)
│   └── data/
│       └── csv_repository.py     # Pandas CSV loader
│
├── api/                           # FastAPI delivery layer
│   ├── routers/
│   │   ├── health.py              # GET  /health
│   │   ├── query.py               # POST /query
│   │   └── tags.py                # GET  /tags
│   ├── schemas/
│   │   └── query.py               # Pydantic request/response models
│   └── dependencies/
│       └── injection.py           # FastAPI Depends() wiring
│
├── core/
│   ├── settings.py                # pydantic-settings (env vars / .env)
│   └── container.py               # Singleton DI container + lifespan hook
│
├── tests/
│   ├── test_query_use_case.py     # Unit tests (all infra mocked)
│   └── test_api.py                # Integration tests (TestClient)
│
├── main.py                        # FastAPI app factory
├── run.py                         # Uvicorn entry point (Lightning AI)
├── requirements.txt
└── .env.example
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Add your dataset
Place your CSV at the path configured by `CSV_PATH` (default: `data/mental_with_multilabels.csv`).  
Required columns: `Concern`, `Suggestion`, `tag`, `multi_labels`, `title`.

### 3. Configure environment
```bash
cp .env.example .env
# Edit .env as needed
```

### 4. Run locally
```bash
python run.py
# or
uvicorn main:app --host 0.0.0.0 --port 7860 --reload
```

API docs: http://localhost:7860/docs

---

## Lightning AI Deployment

1. Push this project to a Lightning AI Studio.
2. Set environment variables in the Studio secrets panel (matches `.env.example`).
3. Set the **run command** to:
   ```
   python run.py
   ```
4. Lightning AI will expose port `7860` automatically.

**Tip:** Set `USE_FALLBACK_LLM=True` for a fast cold start during testing.  
Switch back to `False` once the full environment is validated.

---

## API Endpoints

| Method | Path      | Description                        |
|--------|-----------|------------------------------------|
| GET    | `/health` | Liveness check                     |
| GET    | `/tags`   | List valid `tag_filter` values     |
| POST   | `/query`  | Run the RAG pipeline               |

### POST `/query` — Request body
```json
{
  "query": "I feel overwhelmed and anxious all the time",
  "tag_filter": "All",
  "n_tips": 5
}
```

### POST `/query` — Response
```json
{
  "answer": "**Suggestions for you**\n\n- ...",
  "sources": [
    {
      "concern": "...",
      "title": "...",
      "tag": "anxiety",
      "all_labels": "coping_strategy,personal_experience",
      "suggestions": ["...", "..."]
    }
  ]
}
```

---

## Running Tests
```bash
pytest tests/ -v
```

---

## Swapping Models

Because all ML components implement domain interfaces (ABCs), you can swap any backend without touching use-case code:

- **Different retriever** → implement `EmbeddingRetriever`, register in `container.py`
- **Different reranker** → implement `Reranker`
- **Different LLM** → implement `LLMPostProcessor`
- **Different data source** → implement `SuggestionRepository` (e.g. PostgreSQL, HuggingFace Datasets)
