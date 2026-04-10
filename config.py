"""
config.py – Centralised settings loaded from environment variables.
"""
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Application ──────────────────────────────────────────
    app_name: str = "Docognix API"
    app_version: str = "1.0.0"
    debug: bool = False
    api_prefix: str = "/api/v1"

    # ── CORS ─────────────────────────────────────────────────
    cors_origins: list[str] = [
        "http://localhost:5173",   # Vite dev server
        "http://localhost:3000",
        "https://docognix.vercel.app",
    ]

    # ── Supabase / PostgreSQL (pgvector) ──────────────────────
    supabase_db_url: str = ""   # asyncpg DSN:  postgresql+asyncpg://...
    # e.g. postgresql://postgres:<pw>@db.<ref>.supabase.co:5432/postgres

    # ── Upstash Redis ─────────────────────────────────────────
    upstash_redis_url: str = ""     # rediss://:<token>@<host>:6379
    upstash_redis_token: str = ""   # used for REST API fallback

    # ── Groq (LLM) ───────────────────────────────────────────
    groq_api_key: str = ""
    # Best free quality: llama-3.3-70b-versatile
    # Fast:             llama-3.1-8b-instant
    groq_model: str = "llama-3.3-70b-versatile"
    groq_fallback_model: str = "llama-3.1-8b-instant"

    # ── Embedding model (local, free) ─────────────────────────
    # BAAI/bge-small-en-v1.5  →  384-dim, ~130 MB, very fast
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384

    # ── RAG Pipeline ─────────────────────────────────────────
    chunk_size: int = 512           # target tokens per chunk
    chunk_overlap: int = 64         # overlap tokens between chunks
    retrieval_top_k: int = 20       # candidates from vector search
    rerank_top_n: int = 6           # chunks kept after re-ranking
    hyde_enabled: bool = True       # Hypothetical Document Embeddings
    hybrid_alpha: float = 0.7       # weight: vector vs keyword (1=all-vector)
    max_context_tokens: int = 6000  # hard cap on context sent to LLM
    llm_temperature: float = 0.1
    llm_max_tokens: int = 1024

    # ── Semantic Cache ────────────────────────────────────────
    cache_ttl_seconds: int = 3600   # 1 hour
    cache_similarity_threshold: float = 0.93   # cosine sim to count as HIT

    # ── File Upload ───────────────────────────────────────────
    max_file_size_mb: int = 50
    allowed_extensions: set[str] = {"pdf", "docx", "txt"}

    # ── Confidence Thresholds ─────────────────────────────────
    confidence_high: float = 0.78
    confidence_medium: float = 0.58


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
