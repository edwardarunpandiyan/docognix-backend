-- ============================================================
-- Docognix Database Schema
-- PostgreSQL + pgvector on Supabase
-- Run this in Supabase SQL Editor (fresh database)
-- ============================================================

-- Enable extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================================
-- Conversations Table
-- (frontend uses conversation_id, not session_id)
--
-- Identity model:
--   anonymous_id  = browser identity from localStorage ("br_7xk2m9p")
--                   always present, set before any login
--   user_id       = null until user logs in (future auth)
--
-- On login: all conversations where anonymous_id = "br_7xk2m9p"
--           are reassigned to user_id = "user_abc123"
-- ============================================================
CREATE TABLE IF NOT EXISTS conversations (
    id            UUID        PRIMARY KEY DEFAULT uuid_generate_v4(),
    title         TEXT        NOT NULL DEFAULT 'New Chat',
    anonymous_id  TEXT,                        -- browser identity from localStorage
    user_id       TEXT        DEFAULT NULL,    -- null until login
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_conversations_anonymous_id ON conversations(anonymous_id);
CREATE INDEX IF NOT EXISTS idx_conversations_user_id      ON conversations(user_id);

-- ============================================================
-- Documents Table
-- ============================================================
CREATE TABLE IF NOT EXISTS documents (
    id              UUID        PRIMARY KEY DEFAULT uuid_generate_v4(),
    conversation_id UUID        NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    filename        TEXT        NOT NULL,
    original_name   TEXT        NOT NULL,
    file_type       TEXT        NOT NULL CHECK (file_type IN ('pdf', 'docx', 'txt')),
    file_size       BIGINT      NOT NULL DEFAULT 0,
    page_count      INTEGER,                   -- total pages (PDF/DOCX)
    word_count      INTEGER,                   -- total words
    chunk_count     INTEGER     DEFAULT 0,     -- number of chunks created
    status          TEXT        NOT NULL DEFAULT 'processing'
                                CHECK (status IN ('processing', 'ready', 'error')),
    error_message   TEXT,
    metadata        JSONB       DEFAULT '{}',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_documents_conversation_id ON documents(conversation_id);
CREATE INDEX IF NOT EXISTS idx_documents_status          ON documents(status);

-- ============================================================
-- Chunks Table (with vector embedding)
-- BAAI/bge-small-en-v1.5 → 384 dimensions
-- ============================================================
CREATE TABLE IF NOT EXISTS chunks (
    id              UUID        PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id     UUID        NOT NULL REFERENCES documents(id)     ON DELETE CASCADE,
    conversation_id UUID        NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    content         TEXT        NOT NULL,      -- raw chunk text
    content_lower   TEXT,                      -- lowercased for keyword search
    page_number     INTEGER,                   -- 1-based page the chunk starts on
    page_end        INTEGER,                   -- page the chunk ends on
    chunk_index     INTEGER     NOT NULL,      -- position within document (0-based)
    char_start      INTEGER,                   -- character offset in original document
    char_end        INTEGER,
    token_count     INTEGER,                   -- approximate token count
    embedding       vector(384),               -- dense vector embedding
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Vector similarity index (IVFFlat)
CREATE INDEX IF NOT EXISTS idx_chunks_embedding
    ON chunks USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

CREATE INDEX IF NOT EXISTS idx_chunks_document_id     ON chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_conversation_id ON chunks(conversation_id);
CREATE INDEX IF NOT EXISTS idx_chunks_page_number     ON chunks(page_number);

-- Full-text index for keyword search
CREATE INDEX IF NOT EXISTS idx_chunks_content_fts
    ON chunks USING gin(to_tsvector('english', content));

-- ============================================================
-- Messages Table
-- ============================================================
CREATE TABLE IF NOT EXISTS messages (
    id                UUID        PRIMARY KEY DEFAULT uuid_generate_v4(),
    conversation_id   UUID        NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role              TEXT        NOT NULL CHECK (role IN ('user', 'assistant')),
    content           TEXT        NOT NULL,
    sources           JSONB       DEFAULT '[]',  -- SourceReference[]
    confidence        TEXT        CHECK (confidence IN ('high', 'medium', 'low', NULL)),
    model             TEXT,
    prompt_tokens     INTEGER,
    completion_tokens INTEGER,
    latency_ms        INTEGER,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_messages_conversation_id ON messages(conversation_id);
CREATE INDEX IF NOT EXISTS idx_messages_created_at      ON messages(created_at);

-- ============================================================
-- Auto-update updated_at
-- ============================================================
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE TRIGGER trg_conversations_updated_at
    BEFORE UPDATE ON conversations
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE OR REPLACE TRIGGER trg_documents_updated_at
    BEFORE UPDATE ON documents
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- ============================================================
-- conversation_overview view
-- Used by GET /conversations to return counts in one query
-- ============================================================
CREATE OR REPLACE VIEW conversation_overview AS
SELECT
    c.id,
    c.title,
    c.anonymous_id,
    c.user_id,
    c.created_at,
    c.updated_at,
    COUNT(DISTINCT d.id)              AS document_count,
    COALESCE(SUM(d.chunk_count), 0)   AS total_chunks,
    COUNT(DISTINCT m.id)              AS message_count,
    MAX(m.created_at)                 AS last_message_at
FROM conversations c
LEFT JOIN documents d ON d.conversation_id = c.id AND d.status = 'ready'
LEFT JOIN messages  m ON m.conversation_id = c.id
GROUP BY c.id, c.title, c.anonymous_id, c.user_id, c.created_at, c.updated_at;

-- ============================================================
-- Verify: check all tables created
-- ============================================================
SELECT table_name
FROM information_schema.tables
WHERE table_schema = 'public'
ORDER BY table_name;
