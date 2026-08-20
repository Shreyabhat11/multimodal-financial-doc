# Setup, Run & Deploy Guide

Practical, checklist-style companion to `README.md` and `ARCHITECTURE.md`. This file
answers two questions: **"how do I actually run this"** and **"what do I need to
change before it works with my own credentials/infrastructure."**

Nothing in the codebase needs to change to make it *run* — it's fully wired and
tested with fakes/local SQLite. The changes below are what you need for it to run
**for real**, against real models and a real database.

---

## 1. Absolute minimum to run it locally (no Docker, no Postgres)

This gets you a working API + UI in ~5 minutes, using SQLite instead of Postgres and
the Hugging Face Inference API instead of a local GPU.

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt

cp .env.example .env
```

Now edit `.env` — **only these three lines need real values** for a minimal run:

```env
DATABASE_URL=sqlite:///./local.db
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx        # https://huggingface.co/settings/tokens
ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxx  # https://console.anthropic.com/settings/keys
```

Then:

```bash
# Create the schema (SQLite path — Alembic works against it directly, same as Postgres)
alembic upgrade head

# Terminal 1
uvicorn app.main:app --reload --port 8000

# Terminal 2
streamlit run frontend/streamlit_app.py
```

Open `http://localhost:8501`, upload a PDF from `data/samples/`, and watch it
process. API docs: `http://localhost:8000/docs`.

**Sanity check that everything's wired correctly before you upload real documents:**

```bash
pytest   # should show "81 passed"
```

---

## 2. Running with Docker (recommended for anything beyond a quick local check)

```bash
cp .env.example .env
```

Edit `.env` — at minimum:

```env
ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxx
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
SECRET_KEY=<a real random 64-char string — see Section 3, item 1>
```

`DATABASE_URL` in `.env` is **overridden automatically** by `docker-compose.yml` to
point at the `postgres` service — you don't need to edit it for Docker.

```bash
docker compose up --build
```

This starts Postgres, runs migrations automatically (via
`scripts/docker-entrypoint.sh`), and starts the API (`:8000`) and Streamlit
(`:8501`).

**Local GPU inference instead of the HF Inference API:**

```bash
# in .env: MODEL_PROVIDER=qwen-vl (or llava), MODEL_DEVICE=cuda
docker compose -f docker-compose.yml -f docker-compose.gpu.yml build \
  --build-arg INSTALL_LOCAL_INFERENCE=true api
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up
```

Requires an NVIDIA GPU (see VRAM table in `README.md` §7) and the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
on the Docker host.

---

## 3. What you MUST change before this is safe/correct for real use

These are not stylistic suggestions — using the defaults for these in anything
beyond a local demo is a real security or correctness problem.

### 1. `SECRET_KEY` (`.env`)

```env
SECRET_KEY=dev-only-insecure-secret-key   # ← .env.example default, NEVER use this
```

This key derives the encryption used for account numbers at rest
(`app/database/encryption.py`). Generate a real one:

```bash
python3 -c "import secrets; print(secrets.token_urlsafe(48))"
```

**If you rotate this key after documents are already stored, existing encrypted
account numbers become undecryptable** (`DecryptionError`) — there's no key-rotation
migration built in. If you need key rotation, that's new code to add (decrypt-with-
old-key, re-encrypt-with-new-key, per row) — not present currently.

### 2. `ANTHROPIC_API_KEY` / `HF_TOKEN` (`.env`)

Both are blank placeholders in `.env.example`. Without `ANTHROPIC_API_KEY`, every
document will fail at the `crew_validation` step (caught gracefully — it routes to
`needs_human_review`, doesn't crash — but nothing will ever auto-complete). Without
`HF_TOKEN`, the default `MODEL_PROVIDER=hf-inference` backend cannot call the model
at all.

### 3. `DATABASE_URL` (`.env`, for non-Docker / non-local runs)

```env
DATABASE_URL=postgresql+psycopg2://findoc_user:findoc_pass@localhost:5432/findoc_db
```

`findoc_user`/`findoc_pass` are placeholder credentials. If you're running Postgres
yourself (not via `docker-compose.yml`, which sets its own via `POSTGRES_USER`/
`POSTGRES_PASSWORD`/`POSTGRES_DB` env vars — see `docker-compose.yml`), point this at
real credentials for a database you control.

### 4. `API_CORS_ORIGINS` (`.env` / `configs/config.yaml`)

Defaults to `http://localhost:8501` only. If you deploy the frontend anywhere else
(a real domain), add it here or the browser will block API calls with a CORS error:

```env
API_CORS_ORIGINS=https://your-frontend-domain.com,http://localhost:8501
```

### 5. No authentication layer exists — add one before any non-local deployment

Every endpoint in `app/api/routes/` is currently open to anyone who can reach the
API. This is flagged explicitly in `README.md` §18 (Limitations) — it is a real,
known gap, not an oversight to discover later. Before deploying anywhere reachable
by the public internet:

- Add an API key or OAuth2 dependency to `app/api/routes/documents.py`'s routes
  (FastAPI's `Depends()` pattern makes this a small, contained change — see
  [FastAPI's security docs](https://fastapi.tiangolo.com/tutorial/security/)).
- Put the API behind a reverse proxy / API gateway that enforces TLS.

---

## 4. What you SHOULD review/change depending on your deployment

Not broken as-is, but the defaults were chosen for a portfolio/demo context and are
worth revisiting for a real deployment.

| What | Where | Why you might change it |
|---|---|---|
| `MODEL_PROVIDER`, `MODEL_NAME` | `.env` | Switch to `qwen-vl`/`llava` + local GPU if you process high volume (API costs add up); or point at a different HF-hosted model |
| `CONFIDENCE_THRESHOLD` | `.env` | Default `0.75` is an untuned prior (see `README.md` §18) — tune against real labeled documents once you have some |
| `MAX_FILE_SIZE_MB`, `MAX_PAGES` | `.env` | Defaults (25MB, 50 pages) are reasonable for statements; raise if you expect larger documents |
| `confidence_weights` | `configs/config.yaml` | Must sum to 1.0 (enforced at startup, `app/core/config.py`) — adjust if a different field matters more for your document types |
| File storage backend | `app/services/file_storage.py` | Currently local disk (`data/raw/`) — swap for S3/GCS/Azure Blob if you run more than one API replica, since local disk isn't shared across instances. The module's interface (`save_uploaded_file`/`load_uploaded_file`/`delete_uploaded_file`) is intentionally narrow so this is a contained rewrite, not a codebase-wide change |
| Migration strategy | `scripts/docker-entrypoint.sh` | Currently runs `alembic upgrade head` on every container start — fine for one instance, but multiple replicas starting simultaneously would race. For multi-replica deployments, run migrations as a separate release/CI step instead and remove the migration line from the entrypoint |
| `AGENT_LLM_PROVIDER` / `AGENT_LLM_MODEL` | `.env` | Defaults to Anthropic (`claude-sonnet-4-6`). CrewAI also supports `openai`/`ollama` — set `OPENAI_API_KEY` and change the provider if you'd rather not depend on Anthropic |

---

## 5. Nothing else needs code changes to run

To be explicit about what's already handled and doesn't need touching:

- Database schema/migrations — already generated and tested (`migrations/versions/`)
- All prompts (`app/extraction/prompts.py`) — generic, work for any bank/credit-card/
  loan statement format; no per-bank customization needed to get started
- Validation thresholds and logic — reasonable defaults, all configurable via `.env`
  without touching Python
- CORS, exception handling, logging — already wired

---

## 6. Post-deployment smoke test

After deploying (Docker or otherwise), confirm the whole chain works end to end:

```bash
curl https://your-api-domain/health
# {"status": "ok", ...}

curl -X POST https://your-api-domain/documents/upload \
  -F "file=@data/samples/clean_statement.pdf"
# {"document_id": "...", "status": "uploaded", ...}

# wait a few seconds, then:
curl https://your-api-domain/documents/{document_id}/result
# should show status: "completed" and the extracted fields
```

If it hangs at `processing` indefinitely, check (in order): `ANTHROPIC_API_KEY` is
valid (crew_validation will silently route to human review without crashing, but
never complete — check API server logs for `Crew validation unavailable`), `HF_TOKEN`
is valid, and `DATABASE_URL` is reachable from the API container.
