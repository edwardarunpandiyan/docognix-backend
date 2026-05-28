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
    supabase_db_url: str = ""   # postgresql://postgres:<pw>@db.<ref>.supabase.co:5432/postgres

    # ── Upstash Redis ─────────────────────────────────────────
    upstash_redis_url: str = ""     # rediss://:<token>@<host>:6379
    upstash_redis_token: str = ""   # used for REST API fallback

    # ── Groq (LLM) ───────────────────────────────────────────
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"
    groq_fallback_model: str = "llama-3.1-8b-instant"

    # ── Model mode ────────────────────────────────────────────
    # true  → load models locally via sentence-transformers (torch required)
    # false → call HuggingFace Inference API (no torch loaded, saves ~400 MB RAM)
    use_local_models: bool = False

    # ── HuggingFace Inference API ─────────────────────────────
    # Required when use_local_models=false.
    # Get a free token at: https://huggingface.co/settings/tokens
    hf_api_token: str = ""

    # ── Embedding model ───────────────────────────────────────
    # BAAI/bge-small-en-v1.5 → 384-dim, ~130 MB locally, free on HF API
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384

    # BGE asymmetric query prefix (do not change unless switching models)
    bge_query_prefix: str = (
        "Represent this sentence for searching relevant passages: "
    )

    # ── Reranker model (cross-encoder) ────────────────────────
    # cross-encoder/ms-marco-MiniLM-L-6-v2 → ~80 MB locally, free on HF API
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # ── RAG Pipeline ─────────────────────────────────────────
    chunk_size: int = 300           # target tokens per chunk
    chunk_overlap: int = 40         # overlap tokens between chunks
    retrieval_top_k: int = 20       # Stage 1: candidates fetched by bi-encoder
    rerank_top_n: int = 6           # Stage 1→2: top-N passed to cross-encoder
    rerank_min_ratio: float = 0.40  # prune chunks below 40 % of top CE score
    hyde_enabled: bool = True       # Hypothetical Document Embeddings
    hybrid_alpha: float = 0.7       # weight: vector vs keyword (1=all-vector)
    max_context_tokens: int = 6000  # hard cap on context sent to LLM
    llm_temperature: float = 0.1
    llm_max_tokens: int = 2048

    # ── Semantic Cache ────────────────────────────────────────
    cache_ttl_seconds: int = 3600
    cache_similarity_threshold: float = 0.93

    # ── File Upload ───────────────────────────────────────────
    max_file_size_mb: int = 50
    allowed_extensions: set[str] = {"pdf", "docx", "txt"}

    # ── Confidence Thresholds ─────────────────────────────────
    confidence_high: float = 0.70
    confidence_medium: float = 0.35


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
