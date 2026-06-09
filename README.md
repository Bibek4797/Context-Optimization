# Context Optimization Engine

Stage 1 local-first web app for Python codebase context optimization.

## What Works In Stage 1

- Upload a zipped Python repo or import a GitHub repo URL.
- Parse Python files with Tree-sitter.
- Build a local CodeGraph from Python `ast` relationships.
- Run Graphify when the `graphify` CLI is available, otherwise store an explicit fallback graph derived from CodeGraph.
- Estimate token usage for raw repo text, chunk retrieval context, CodeGraph context, Graphify context, merged context, and per-query LLM usage.
- Ask Standard Repo QA questions with Gemini.
- Ask Graph-Optimized Repo QA questions using CodeGraph and Graphify contexts together.
- Compare baseline vs optimized context tokens for the same query.

## Fast Public Demo With Streamlit

This is the fastest path for showing the project to mentors. It reuses the same backend analysis services without requiring the Next.js frontend or a separate FastAPI server.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run streamlit_app.py
```

For Gemini locally, either create `.env` from `.env.example` or create `.streamlit/secrets.toml` from `.streamlit/secrets.toml.example`.

For Streamlit Community Cloud:

1. Push this repo to GitHub.
2. Create a Streamlit app from the repo.
3. Set the entrypoint file to `streamlit_app.py`.
4. Paste your Gemini key into the app secrets:

```toml
GEMINI_API_KEY = "your_key_here"
GEMINI_MODEL = "gemini-2.5-flash"
```

Streamlit Cloud stores secrets outside the repository, which is the right place for the Gemini key.

## Full Local Dev Setup

```powershell
Copy-Item .env.example .env
```

Add your Gemini key to `.env`:

```text
GEMINI_API_KEY=your_key_here
```

Backend:

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Frontend:

```powershell
cd frontend
npm install
npm run dev
```

Open `http://localhost:3000`.

The frontend defaults to `http://localhost:8000/api`. To override it, copy `frontend/.env.local.example` to `frontend/.env.local`.

## Optional Graphify

Graphify is optional in Stage 1. If the `graphify` command is available, the backend runs it and normalizes `graph.json`. If it is missing or fails, the app stores:

- raw CLI/fallback details under `data/state/repos/{repo_id}/graphify_raw/`
- a `graphify-fallback` graph clearly marked as fallback
- warnings surfaced in repo status and logs

The fallback is not presented as native Graphify output.

## Architecture

```text
backend/
  app/api/              FastAPI endpoints
  app/core/             config and singleton service wiring
  app/models/           Pydantic API/storage schemas
  app/services/         ingestion, parsing, graph, retrieval, token, LLM services
frontend/
  app/                  Next.js app router
  components/           reusable UI pieces
  lib/                  API client and frontend types
data/
  repos/                local imported source trees
  uploads/              uploaded zip archives
  state/                JSON analysis artifacts and query records
  logs/                 reserved for app-level logs
```

## Library-Based vs Custom

- Tree-sitter parsing uses `tree-sitter` and `tree-sitter-python`.
- Token estimation uses `tiktoken`; Gemini prompt/response counts are exact when Gemini returns usage metadata.
- Gemini uses the official `google-genai` SDK through an environment-based provider adapter.
- CodeGraph is a custom Stage 1 Python static analyzer built on Python `ast`.
- Graphify is an optional external CLI adapter with an honest fallback.
- Retrieval is custom lexical ranking for Stage 1, designed to be replaced by embeddings or hybrid search later.

## Known Limitations

- Python-only repository support.
- Static call resolution is approximate for dynamic Python.
- No background job queue yet; ingestion runs synchronously.
- Token counts outside Gemini calls are estimates, labeled as such.
- Graphify native output depends on a locally installed `graphify` CLI.
- Local JSON storage is intended for Stage 1, not multi-user production.


