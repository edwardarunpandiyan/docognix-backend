# Docognix Backend

RAG-powered document QA API built with **FastAPI**, **pgvector on Supabase**, **Upstash Redis**, **Groq LLM**, and local **BAAI/bge-small-en-v1.5** embeddings.

---

## Architecture

```
                 ┌─────────────────────────────────────────────────┐
                 │              Docognix React Frontend             │
                 └──────────────────┬──────────────────────────────┘
                                    │ HTTP / SSE
                 ┌──────────────────▼──────────────────────────────┐
                 │           FastAPI Application (main.py)          │
                 │  /api/v1/sessions  /api/v1/documents  /api/v1/chat│
                 └──────┬───────────────┬──────────────────┬────────┘
                        │               │                  │
              ┌─────────▼──────┐ ┌──────▼──────┐  ┌───────▼───────┐
              │  Supabase      │ │  Upstash     │  │  Groq API     │
              │  PostgreSQL    │ │  Redis       │  │  (LLM)        │
              │  + pgvector    │ │  (Sem.Cache) │  │  llama-3.3-70b│
              └───────────────-┘ └─────────────┘  └───────────────┘
                        │
              ┌─────────▼──────┐
              │ BAAI/bge-small │
              │ -en-v1.5       │
              │ (local embed.) │
              └────────────────┘
```

### RAG Pipeline (per query)

```
User Query
  │
  ├─► Semantic Cache Check (Redis)  ──HIT──► Stream cached answer
  │
  └─► MISS ──►
        │
        ├─ HyDE: Generate hypothetical answer (llama-3.1-8b-instant)
        ├─ Embed: query + HyDE answer  →  384-dim vector
        ├─ Vector Search: pgvector cosine top-20
        ├─ Keyword Search: PostgreSQL tsvector top-20
        ├─ RRF Fusion: Combine + re-rank → top-6 chunks
        ├─ Build Context: inject page refs, truncate to 6000 tokens
        ├─ Load Chat History: last 6 turns
        ├─ Stream: llama-3.3-70b-versatile via Groq SSE
        ├─ Persist: user + assistant messages in DB
        └─ Cache: store result in Redis semantic cache
```

---

## Quick Start

### Prerequisites
- Python 3.12+
- A [Supabase](https://supabase.com) project (free tier works)
- An [Upstash Redis](https://upstash.com) database (free tier works)
- A [Groq](https://console.groq.com) API key (free)

### 1. Clone & Install

```bash
cd docognix-backend
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# CPU-only torch (saves ~2 GB vs CUDA build)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your actual credentials
```

**Required values in `.env`:**

| Variable | Where to get it |
|---|---|
| `SUPABASE_DB_URL` | Supabase → Settings → Database → Connection string (URI) |
| `UPSTASH_REDIS_URL` | Upstash Console → Database → Details → Redis URL |
| `GROQ_API_KEY` | console.groq.com → API Keys |

### 3. Run Database Migrations

Open your Supabase project's SQL editor and run:

```sql
-- database/schema.sql
```

Or use the Supabase CLI:
```bash
supabase db push --db-url "$SUPABASE_DB_URL" < database/schema.sql
```

### 4. Start the Server

```bash
uvicorn main:app --reload --port 8000
```

API docs: http://localhost:8000/docs

---

## API Reference

### Sessions

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/v1/sessions` | Create a new chat session |
| `GET` | `/api/v1/sessions` | List all sessions |
| `GET` | `/api/v1/sessions/{id}` | Get session details |
| `PATCH` | `/api/v1/sessions/{id}` | Rename session |
| `DELETE` | `/api/v1/sessions/{id}` | Delete session + all data |

### Documents

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/v1/sessions/{id}/documents` | Upload & ingest a file |
| `GET` | `/api/v1/sessions/{id}/documents` | List documents in session |
| `GET` | `/api/v1/sessions/{id}/documents/{docId}` | Get document details |
| `GET` | `/api/v1/sessions/{id}/documents/{docId}/status` | Poll ingestion status |
| `DELETE` | `/api/v1/sessions/{id}/documents/{docId}` | Delete document |

### Chat

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/v1/chat/{session_id}` | Send message → SSE stream |
| `GET` | `/api/v1/chat/{session_id}/messages` | Get message history |
| `DELETE` | `/api/v1/chat/{session_id}/messages` | Clear history |

---

## SSE Event Stream Format

The frontend connects to `POST /api/v1/chat/{session_id}` and handles these events:

```
event: meta
data: {"type":"meta","session_id":"...","user_message_id":"..."}

event: sources
data: {"type":"sources","sources":[{"chunk_id":"...","document_name":"...","content":"...","page_number":3,"similarity_score":0.87,"confidence":"high",...}]}

event: token
data: {"type":"token","content":"Based on page 3 of "}

event: token
data: {"type":"token","content":"the contract..."}

event: done
data: {"type":"done","message_id":"...","confidence":"high","prompt_tokens":1200,"completion_tokens":180,"latency_ms":1340}
```

### Frontend SSE listener (reference)

```javascript
const response = await fetch(`/api/v1/chat/${sessionId}`, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ message: userQuery, stream: true }),
});

const reader = response.body.getReader();
const decoder = new TextDecoder();
let buffer = '';

while (true) {
  const { done, value } = await reader.read();
  if (done) break;
  buffer += decoder.decode(value, { stream: true });
  const lines = buffer.split('\n');
  buffer = lines.pop();   // keep incomplete line

  for (const line of lines) {
    if (line.startsWith('data: ')) {
      const event = JSON.parse(line.slice(6));
      if (event.type === 'token')   appendToken(event.content);
      if (event.type === 'sources') setSources(event.sources);
      if (event.type === 'done')    finalize(event);
      if (event.type === 'error')   showError(event.message);
    }
  }
}
```

---

## Key Design Decisions

### Model Selection (Free Tier)
| Component | Model | Why |
|-----------|-------|-----|
| **LLM** | `llama-3.3-70b-versatile` (Groq) | Best free quality; 128K context; Groq's inference is extremely fast |
| **LLM Fallback** | `llama-3.1-8b-instant` | Used for HyDE; ~10× faster, still good |
| **Embeddings** | `BAAI/bge-small-en-v1.5` | 384-dim, ~130 MB, runs on CPU in <20ms, top MTEB scores |

### Intelligence Features
- **HyDE** (Hypothetical Document Embeddings) – dramatically improves recall for ambiguous queries
- **Hybrid Search** – vector + BM25 keyword, combined via Reciprocal Rank Fusion
- **Semantic Cache** – cosine similarity matching (threshold 0.93) avoids redundant LLM calls
- **Overlapping Chunks** – 64-token overlap prevents context loss at chunk boundaries
- **Page-aware Chunking** – chunks carry `page_number`/`page_end` for accurate citation highlighting
- **Chat History** – last 6 turns injected into LLM context for multi-turn coherence
- **Confidence Scoring** – high/medium/low based on cosine similarity of top retrieved chunk
- **Automatic Fallback** – if primary LLM fails, falls back to fast model seamlessly

### Data Stored Per Chunk (for Frontend)
```json
{
  "chunk_id": "uuid",
  "document_id": "uuid",
  "document_name": "contract.pdf",
  "content": "The tenant shall...",
  "page_number": 3,
  "page_end": 3,
  "chunk_index": 12,
  "similarity_score": 0.872,
  "keyword_score": 0.14,
  "combined_score": 0.00821,
  "confidence": "high"
}
```

---

## Production Deployment

### Railway / Render / Fly.io

```bash
# Set environment variables in your platform dashboard
# Then deploy:
docker build -t docognix-api .
docker run -p 8000:8000 --env-file .env docognix-api
```

### Supabase Connection Pooling
For production, use Supabase's **Supavisor** connection pooler (port 6543, transaction mode):
```
postgresql://postgres.<project-ref>:<password>@aws-0-<region>.pooler.supabase.com:6543/postgres
```

---

## Project Structure

```
docognix-backend/
├── main.py                    # FastAPI app, lifespan, middleware
├── config.py                  # Settings (pydantic-settings + .env)
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .env.example
├── database/
│   ├── schema.sql             # Full PostgreSQL schema with pgvector
│   ├── postgres.py            # asyncpg pool + vector codec
│   └── redis_client.py        # Upstash Redis async client
├── models/
│   ├── documents.py           # Upload, chunk, source schemas
│   └── chat.py                # Session, message, SSE event schemas
├── services/
│   ├── embedding.py           # BGE-small local embeddings
│   ├── document_processor.py  # Parse PDF/DOCX/TXT → chunk → embed → store
│   ├── retrieval.py           # Hybrid retrieval (vector + keyword + RRF + HyDE)
│   ├── rag.py                 # Full RAG pipeline with SSE streaming
│   └── cache.py               # Semantic cache (Redis)
├── routers/
│   ├── sessions.py            # /api/v1/sessions
│   ├── documents.py           # /api/v1/sessions/{id}/documents
│   └── chat.py                # /api/v1/chat
└── utils/
    └── text_utils.py          # Chunking, token counting, RRF, BM25
```
