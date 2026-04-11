-- ============================================================
-- Docognix Database Schema
-- PostgreSQL + pgvector on Supabase
-- ============================================================

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================================
-- Sessions Table
-- ============================================================
CREATE TABLE IF NOT EXISTS sessions (
    id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    title       TEXT NOT NULL DEFAULT 'New Chat',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ============================================================
-- Documents Table
-- ============================================================
CREATE TABLE IF NOT EXISTS documents (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id      UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    filename        TEXT NOT NULL,
    original_name   TEXT NOT NULL,
    file_type       TEXT NOT NULL CHECK (file_type IN ('pdf', 'docx', 'txt')),
    file_size       BIGINT NOT NULL DEFAULT 0,
    page_count      INTEGER,          -- total pages (PDF/DOCX)
    word_count      INTEGER,          -- total words
    chunk_count     INTEGER DEFAULT 0,-- number of chunks created
    status          TEXT NOT NULL DEFAULT 'processing'
                    CHECK (status IN ('processing', 'ready', 'error')),
    error_message   TEXT,
    metadata        JSONB DEFAULT '{}',-- extra doc-level metadata
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_documents_session_id ON documents(session_id);
CREATE INDEX IF NOT EXISTS idx_documents_status     ON documents(status);

-- ============================================================
-- Chunks Table (with vector embedding)
-- BAAI/bge-small-en-v1.5 → 384 dimensions
-- ============================================================
CREATE TABLE IF NOT EXISTS chunks (
    id            UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id   UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    session_id    UUID NOT NULL REFERENCES sessions(id)  ON DELETE CASCADE,
    content       TEXT NOT NULL,       -- raw text of the chunk
    content_lower TEXT,                -- lower-cased for BM25 keyword search
    page_number   INTEGER,             -- 1-based page the chunk starts on
    page_end      INTEGER,             -- page the chunk ends on (multi-page)
    chunk_index   INTEGER NOT NULL,    -- position within document (0-based)
    char_start    INTEGER,             -- char offset in original document
    char_end      INTEGER,
    token_count   INTEGER,             -- approximate token count
    embedding     vector(384),         -- dense embedding
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Vector similarity index (IVFFlat – good balance of speed / recall)
CREATE INDEX IF NOT EXISTS idx_chunks_embedding
    ON chunks USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

-- Covering indexes for fast filtered searches
CREATE INDEX IF NOT EXISTS idx_chunks_document_id  ON chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_session_id   ON chunks(session_id);
CREATE INDEX IF NOT EXISTS idx_chunks_page_number  ON chunks(page_number);

-- Full-text index for BM25-style keyword search
CREATE INDEX IF NOT EXISTS idx_chunks_content_fts
    ON chunks USING gin(to_tsvector('english', content));

-- ============================================================
-- Messages Table
-- ============================================================
CREATE TABLE IF NOT EXISTS messages (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id      UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    role            TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content         TEXT NOT NULL,
    sources         JSONB DEFAULT '[]',  -- array of SourceReference objects
    confidence      TEXT CHECK (confidence IN ('high', 'medium', 'low', NULL)),
    model           TEXT,                -- which LLM was used
    prompt_tokens   INTEGER,
    completion_tokens INTEGER,
    latency_ms      INTEGER,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_messages_session_id  ON messages(session_id);
CREATE INDEX IF NOT EXISTS idx_messages_created_at  ON messages(created_at);

-- ============================================================
-- Helper function: update updated_at automatically
-- ============================================================
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE TRIGGER trg_sessions_updated_at
    BEFORE UPDATE ON sessions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE OR REPLACE TRIGGER trg_documents_updated_at
    BEFORE UPDATE ON documents
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- ============================================================
-- Useful views
-- ============================================================

-- Session overview with document and message counts
CREATE OR REPLACE VIEW session_overview AS
SELECT
    s.id,
    s.title,
    s.created_at,
    s.updated_at,
    COUNT(DISTINCT d.id)                     AS document_count,
    COALESCE(SUM(d.chunk_count), 0)          AS total_chunks,
    COUNT(DISTINCT m.id)                     AS message_count,
    MAX(m.created_at)                        AS last_message_at
FROM sessions s
LEFT JOIN documents d ON d.session_id = s.id AND d.status = 'ready'
LEFT JOIN messages  m ON m.session_id = s.id
GROUP BY s.id, s.title, s.created_at, s.updated_at;
