# Helix SROP — Stateful RAG Orchestration Pipeline

AI Support Concierge that routes user queries between a **KnowledgeAgent** (RAG-backed documentation search), **AccountAgent** (internal data lookups), and **EscalationAgent** (support ticket creation), with state that survives process restarts.

---

## Quick Start

```bash
# 1. Create virtual environment and install dependencies
uv sync --extra dev        # or: pip install -e ".[dev]"

# 2. Set up environment variables
cp .env.example .env
# Edit .env → set GOOGLE_API_KEY=your-key

# 3. Ingest documentation into the vector store
python -m app.rag.ingest --path docs/

# 4. Run the server
uvicorn app.main:app --reload

# 5. Run tests
pytest -q
```

### Docker

```bash
cp .env.example .env       # set GOOGLE_API_KEY
docker compose up --build
```

---

## Architecture

```
POST /v1/chat/{session_id}
         │
         ▼
┌─────────────────────────┐
│  SROP Pipeline          │
│  1. Check idempotency   │  ← E1
│  2. Load session state  │
│  3. Build dynamic agent │
│  4. Run ADK orchestrator│
│  5. Collect events/trace│
│  6. Save updated state  │
│  7. Write trace to DB   │
│  8. Persist tickets     │  ← E2
└────────────┬────────────┘
             │ routes via ADK AgentTool
       ┌─────┼──────────┐
       ▼     ▼          ▼
 Knowledge  Account  Escalation    ← E2
 Agent      Agent    Agent
 (RAG)      (DB)     (tickets)
       │
  ┌────┴────┐
  ChromaDB  Reranker             ← E4
  (vectors) (LLM judge)
```

### API Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/v1/sessions` | Create session with `{user_id, plan_tier}` |
| `POST` | `/v1/chat/{session_id}` | Send message, get `{reply, routed_to, trace_id}` |
| `GET`  | `/v1/traces/{trace_id}` | Retrieve structured trace for debugging |
| `GET`  | `/healthz` | Health check |

#### Special Headers

| Header | Effect |
|--------|--------|
| `Idempotency-Key: <key>` | E1: Replay returns original response |
| `Accept: text/event-stream` | E3: Stream response via SSE |

---

## State Persistence — Design Decision

**Chosen: Pattern 3 — Inject SessionState into instruction per turn.**

On each turn, `SessionState` (user_id, plan_tier, last_agent, turn_count, last_ticket_id) is loaded from the SQLite `sessions.state` JSON column and injected into the root agent's system instruction. The agent is built fresh per turn with this context.

### Tradeoffs

| Approach | Pros | Cons |
|----------|------|------|
| **Pattern 3 (chosen)** | Simplest; no custom SessionService; state survives restarts; small context window cost | Must rebuild agent per turn; no full message history in LLM context |
| Pattern 1 (full session in DB) | Full history available to LLM | Complex; must implement `BaseSessionService`; large context windows |
| Pattern 2 (re-hydrate history) | Rich context per turn | Token cost grows linearly with conversation length |

Pattern 3 is optimal for this use case: the state is small (<200 bytes), and the LLM only needs to know the user's plan tier and which agent last ran — not the full conversation history.

---

## Chunking Strategy

**Heading-aware chunking** (split on `##` / `###` markdown headings).

The Helix documentation files are structured Markdown with clear heading hierarchies. Splitting on headings keeps each chunk as a coherent section (e.g., "How to rotate a deploy key" stays in one chunk rather than being split mid-explanation).

For sections exceeding `chunk_size` (default 512 chars), sentence-aware sub-chunking is applied with 1-sentence overlap to preserve context at boundaries.

**Stable IDs:** `chunk_` + `sha256(file_path::chunk_index)[:16]` — deterministic, so re-ingest deduplicates via ChromaDB's `upsert`.

---

## Extensions Implemented (30 pts)

### E1 — Idempotency (6 pts)
`Idempotency-Key` header on `POST /v1/chat/{session_id}`. If the same key is sent again, the stored response is returned without re-running the pipeline. Implemented via the `idempotency_records` table. Prevents duplicate processing of retried requests.

### E2 — Escalation Agent (5 pts)
Third sub-agent (`escalation_agent`) with `create_ticket(user_id, summary, priority)` tool. Writes to the `tickets` table and returns a ticket ID (`TKT-XXXXXXXX`). The ticket ID is stored in `SessionState.last_ticket_id` and available in follow-up turns.

### E3 — Streaming SSE (5 pts)
Send `Accept: text/event-stream` with your chat request to receive Server-Sent Events. Events include `tool_call`, `text_chunk`, and `done` with the final `routed_to` and `trace_id`.

### E4 — Reranking (4 pts)
LLM-as-judge reranker using Gemini. After vector search returns initial results, each chunk is scored for relevance by the LLM. Controlled by `RERANK_ENABLED` setting. Falls back gracefully to vector-only ranking on failure. Run `python eval/rerank_eval.py` to see before/after comparison.

### E5 — Guardrails (4 pts)
- **Refusal:** Root orchestrator refuses out-of-scope queries (creative writing, unrelated topics) with "I can only help with Helix product and account questions."
- **PII Redaction:** Emails, phone numbers, SSNs, and credit card numbers are redacted from logs via a structlog processor. Never reaches application logs or trace endpoints.

### E6 — Docker (3 pts)
Multi-stage Dockerfile (builder + runtime). `docker compose up` starts the full stack with persistent volumes, curl-based health check, and auto-restart.

### E7 — Eval Harness (3 pts)
```bash
python eval/run_eval.py --base-url http://127.0.0.1:8000
```
Runs 12 eval queries covering knowledge, account, escalation, and out-of-scope routing. Reports accuracy as a percentage with pass/fail table. Results saved to `eval/eval_results.json`.

---

## Known Limitations

- **Single-session context:** Context window grows per session (state fields only, not full history). No cross-session memory.
- **Mock account data:** `get_recent_builds` and `get_account_status` return seeded mock data. The ADK tool wiring is real.
- **Single embedding model:** Uses Google's `gemini-embedding-001`. Switching models requires re-ingest.
- **Reranking latency:** E4 adds an extra LLM call per search. Can be disabled via `RERANK_ENABLED=false`.

---

## Time Breakdown

| Task | Time |
|------|------|
| Env setup + DB schema + FastAPI boilerplate | 25 min |
| `ingest.py` — chunk + embed + upsert | 30 min |
| `search_docs` tool + vector store wiring | 20 min |
| ADK agents — orchestrator + 3 sub-agents | 35 min |
| `pipeline.py` — state in/out of ADK, trace write | 40 min |
| Routes — session create + chat + trace | 15 min |
| State persistence (Pattern 3 wiring) | 15 min |
| Tests + README | 20 min |
| Extensions (E1–E7) | 60 min |
| **Total** | **~4h 20min** |

---

## Project Structure

```
app/
├── main.py                  # FastAPI app, lifespan, error handlers
├── settings.py              # Pydantic settings from .env
├── agents/
│   ├── orchestrator.py      # Root agent (Pattern 3 builder)
│   ├── knowledge.py         # KnowledgeAgent (RAG)
│   ├── account.py           # AccountAgent (mock tools)
│   ├── escalation.py        # EscalationAgent (E2 — tickets)
│   └── tools/
│       ├── search_docs.py   # Vector store search + reranking (E4)
│       ├── account_tools.py # Mock build/account tools
│       └── escalation_tools.py # create_ticket tool (E2)
├── api/
│   ├── errors.py            # Typed exceptions + RFC 7807
│   ├── routes_sessions.py   # POST /v1/sessions
│   ├── routes_chat.py       # POST /v1/chat/{id} + E1 idempotency + E3 SSE
│   └── routes_traces.py     # GET /v1/traces/{id}
├── db/
│   ├── models.py            # SQLAlchemy models (+ Ticket, IdempotencyRecord)
│   └── session.py           # Async engine + get_db
├── obs/
│   ├── logging.py           # structlog config + PII redaction (E5)
│   └── guardrails.py        # PII redaction utilities (E5)
├── rag/
│   ├── ingest.py            # CLI: chunk → embed → upsert
│   └── reranker.py          # LLM-as-judge reranker (E4)
└── srop/
    ├── pipeline.py          # Core pipeline + streaming (E3)
    └── state.py             # SessionState schema (+ last_ticket_id)
eval/
├── eval_cases.json          # Routing eval test cases (E7)
├── run_eval.py              # Routing accuracy harness (E7)
└── rerank_eval.py           # Reranking before/after comparison (E4)
tests/
├── conftest.py              # Fixtures (mock_adk, client, db)
├── test_api.py              # Integration tests (+ E1, E2, E3, E5)
├── test_guardrails.py       # PII redaction tests (E5)
└── test_retriever.py        # RAG unit tests
```
