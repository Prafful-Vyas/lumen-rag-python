# Lumen (Python)

Multi-tenant semantic document search with cited answers. Teams upload PDFs, Word files and
notes; Lumen runs them through a staged ingestion pipeline, then answers questions with hybrid
retrieval (vector + lexical, fused with Reciprocal Rank Fusion, then reranked) and streams a
grounded, cited answer over Server-Sent Events.

This is a Python implementation of [Lumen](https://github.com/Prafful-Vyas/Lumen), which was
originally written in TypeScript (Express, BullMQ, Next.js). The architecture is the same; every
layer is now Python.

| Layer | Python stack |
|---|---|
| API | FastAPI (async), Pydantic v2, JWT (PyJWT) + bcrypt, SSE via `StreamingResponse` |
| Ingestion workers | arq on Redis: four stage queues, each with its own concurrency and retry policy |
| Storage | MongoDB via PyMongo's async client; Redis for caches, metering and rate limits |
| Retrieval | numpy brute-force cosine + MongoDB `$text`, or Atlas `$vectorSearch`/`$search`; RRF; LLM listwise reranker |
| LLM / embeddings | OpenAI (`text-embedding-3-large` at 1024 dims, `gpt-4o-mini`) or deterministic offline mocks |
| Extraction | pypdf (per page), pypdfium2 + Tesseract OCR for scanned pages, python-docx to Markdown |
| Tokenization | tiktoken (`cl100k_base`) |
| UI | Streamlit (login, upload with live stage progress, streaming Q&A with sources, usage) |
| Delivery | Dockerfile, docker-compose (api, worker, ui, mongo, redis), GitHub Actions CI |

## Architecture

```
            upload                     arq queues (Redis)
 Streamlit ─────────► FastAPI ──► extract ──► chunk ──► embed ──► index ──► chunks go live
    ▲                   │          pypdf/OCR  structural  batched    atomic
    │  SSE: sources,    │          docx→md    chunker +   96/call,   publish gate,
    │  tokens, done     │                     contextual  cached,    corpusVersion++
    │                   │                     headers     rate-limited
    └───────────────────┤
                        ▼
   normalize → (embed query ‖ lexical search) → vector search → RRF → tenant guard → rerank
   → refusal check → prompt (budgeted, de-duplicated, best-last) → stream → validate citations
   → audit record + answer cache + metering
```

`src/lumen/` layout:

- `ports.py`: protocols for everything external (embeddings, completion, reranker, vector index, object storage, job queue)
- `adapters/`: OpenAI + mock providers, LLM + mock rerankers, local/Atlas vector indexes, disk/S3 storage
- `composition.py`: the single place env vars pick adapters; the API, the worker and the eval all build from it
- `ingestion/`: extractors, structural chunker, contextual headers, embed service, staged pipeline, arq worker
- `query/`: pure query logic (`core.py`), Redis answer cache and metering (`cache.py`), retrieval and generation services
- `repositories.py`: MongoDB access, workspace-scoped by construction
- `api/`: FastAPI routes, auth dependencies, rate limiting

## Design decisions

- **Four queues, not one job.** Each stage fails differently, so each has its own policy: extract
  2 attempts with a fixed 5 s delay (a corrupt PDF stays corrupt), chunk 3 attempts, embed 5 with
  exponential backoff (provider 429s), index 5. A document is marked `failed` only once its
  stage's retries are exhausted. Extracted page text is checkpointed on the document, so a
  chunk-stage retry never re-extracts, and a retried chunk stage clears its partial output first.
- **Atomic publish.** Chunks are inserted with `live: false` and flipped together by the index
  stage. If more than 5% of chunks failed to embed, the document is not published at all.
- **Hybrid retrieval with RRF.** Cosine and text scores are on incomparable scales, so fusion is
  by rank (`k = 60`). Every query records each candidate's vector rank, lexical rank, RRF score
  and rerank score for offline analysis.
- **Contextual chunk headers.** Each chunk is embedded as `[Document - Section]\n<text>`, with
  headers generated in batches of 20 using the document outline as context (deterministic
  headers in mock mode).
- **Grounding.** The prompt keeps the top-ranked source next to the question, skips near-duplicate
  chunks (cosine > 0.95) and respects a token budget. Any `[n]` the model invents is stripped and
  counted. Below a relevance threshold Lumen refuses instead of generating.
- **Tenant isolation in layers.** The workspace id comes only from the verified JWT; every
  repository read filters on it; every retrieval result is checked again after fusion.
  Deleting a document delists its chunks immediately.
- **Cost control.** Embeddings are cached in Redis by `sha256(model + text)`, answers are cached by
  workspace + normalized query + filters + `corpusVersion` (bumped on every upload/delete, so a
  cached answer can never ignore a new document), and per-workspace monthly spend is metered
  against a cap that blocks further queries once reached.

## Run it locally

Needs Python 3.11+, MongoDB and Redis. No API key is needed: without `OPENAI_API_KEY` every AI
adapter falls back to a deterministic offline mock.

```bash
python -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[ui,dev]"
cp .env.example .env

python -m lumen.seed                               # demo login: demo@lumen.local / Demo1234!
uvicorn lumen.api.app:app --port 8000              # API, docs at http://localhost:8000/docs
python -m lumen.ingestion.worker                   # ingestion workers
streamlit run ui/streamlit_app.py                  # UI at http://localhost:8501
python scripts/smoke.py                            # upload -> ingest -> ask, end to end
```

Or with Docker: `cp .env.example .env && docker compose up --build`.

Set `OPENAI_API_KEY` for real embeddings, generation, reranking and contextual headers. Set
`VECTOR_BACKEND=atlas` (with `chunks_vector` and `chunks_lexical` search indexes on the `chunks`
collection) and `STORAGE_BACKEND=s3` for the production backends.

## Tests and evaluation

```bash
pytest -q -rs          # 49 tests; integration tests need MongoDB + Redis and skip themselves otherwise
python -m eval.run_eval [--ablate-lexical | --ablate-vector]
```

- **Unit tests** cover RRF fusion, query normalization, citation validation, prompt assembly
  (ordering, de-duplication, budget), the refusal policy, the tenant guard, reranker JSON parsing,
  the structural chunker (table/list atomicity, sentence splitting, overlap), table detection,
  pagination, DOCX-to-Markdown, PDF page extraction, OCR fallback, storage path traversal, the
  token bucket and the retry policies.
- **Integration tests** drive the FastAPI app over ASGI against real MongoDB and Redis, with the
  pipeline run inline using the same stage code and retry policies as the workers. They cover
  signup/login/join and role checks, upload through every stage to a cited SSE answer, the answer
  cache and its invalidation on upload, cross-workspace isolation, delete-delists, PDF and DOCX
  ingestion, a corrupt file failing at extract, the monthly cost cap and the rate limiter.
- **Eval harness** ingests a labeled 5-document, 10-question sample set and reports Recall@5,
  Recall@20, MRR and nDCG@10, with ablations that drop one retrieval path.

## What was verified, and what was not

Verified on Windows 10 with Python 3.12, MongoDB 7.0 and Redis 7.2 (Memurai):

- `pytest`: 49 passed (OCR test run with Tesseract 5 installed); `ruff check` and `ruff format --check` clean.
- Separate processes (uvicorn API, arq worker, Streamlit): `scripts/smoke.py` uploaded a document,
  watched the worker take it through all four stages, and got a streamed, cited answer.
- arq retry behaviour: a corrupt PDF was retried once after 5 s, then marked failed at `extract`.
- The Streamlit UI was driven with Streamlit's `AppTest` against the running API (login,
  documents, workspace usage, ask).
- Eval with mock providers: hybrid MRR 0.833 / nDCG@10 0.876; lexical path removed: MRR 0.297 /
  nDCG@10 0.468. Mock embeddings carry no meaning, so these numbers only show the harness and
  fusion working; on this tiny set Recall@5 is 100% by construction. Real retrieval quality needs
  `OPENAI_API_KEY` and a real labeled set.

Verified in GitHub Actions (Ubuntu): lint, all 49 tests with Tesseract installed, the eval, and a
Docker job that builds the image, boots `api`, `worker`, `mongo` and `redis` with docker compose,
seeds the demo user and passes `scripts/smoke.py` (queued -> extracting -> chunking -> embedding -> ready).

Not verified: the OpenAI adapters against the live API, MongoDB Atlas search indexes and S3.

## Known simplifications

- PDF citations point to a page, not a highlighted span (no bounding boxes).
- DOCX/TXT/MD pages are synthetic (~3000 characters) because those formats have no fixed pages.
- PDF table detection is a plain-text column heuristic; DOCX tables convert exactly.
- Reranking uses an LLM listwise scorer as a stand-in for a cross-encoder, behind the same port.
- The local vector backend is a brute-force scan per workspace; use Atlas beyond a few hundred thousand chunks.
- PPTX is rejected at upload rather than mis-extracted. Uploads are type- and size-checked but not virus-scanned.
- Monthly spend counters live in Redis; the `queries` collection is the record to reconcile against.
