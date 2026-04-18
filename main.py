"""
main.py – Docognix FastAPI application.
"""
from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config import settings
from database.postgres import get_pool, close_pool
from database.redis_client import get_redis, close_redis
from routers import conversations_router, documents_router, chat_router

logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("🚀  Starting Docognix API v%s", settings.app_version)

    log.info("Loading embedding model: %s …", settings.embedding_model)
    from services.embedding import _get_model
    import asyncio
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _get_model)
    log.info("✅  Embedding model ready.")

    try:
        pool = await get_pool()
        async with pool.acquire() as conn:
            version = await conn.fetchval("SELECT version()")
        log.info("✅  PostgreSQL connected: %s", version[:40])
    except Exception as e:
        log.error("❌  PostgreSQL connection failed: %s", e)

    try:
        redis = await get_redis()
        await redis.ping()
        log.info("✅  Redis (Upstash) connected.")
    except Exception as e:
        log.warning("⚠️   Redis connection failed (cache disabled): %s", e)

    yield

    log.info("Shutting down …")
    await close_pool()
    await close_redis()
    log.info("Bye 👋")


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="RAG-powered document QA API. Supports PDF, DOCX, and TXT with SSE streaming.",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Request-Id"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    latency = int((time.time() - start) * 1000)
    log.info("%s %s → %d (%d ms)", request.method, request.url.path, response.status_code, latency)
    response.headers["X-Response-Time-Ms"] = str(latency)
    return response


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    log.exception("Unhandled exception on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal server error occurred.", "type": type(exc).__name__},
    )


PREFIX = settings.api_prefix   # /api/v1

app.include_router(conversations_router, prefix=PREFIX)
app.include_router(documents_router,     prefix=PREFIX)
app.include_router(chat_router,          prefix=PREFIX)


@app.get("/health", tags=["Health"])
async def health():
    return {"status": "ok", "version": settings.app_version}


@app.get("/health/ready", tags=["Health"])
async def readiness():
    checks: dict[str, str] = {}
    try:
        pool = await get_pool()
        async with pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
        checks["postgres"] = "ok"
    except Exception as e:
        checks["postgres"] = f"error: {e}"
    try:
        redis = await get_redis()
        await redis.ping()
        checks["redis"] = "ok"
    except Exception as e:
        checks["redis"] = f"error: {e}"
    overall = "ok" if all(v == "ok" for v in checks.values()) else "degraded"
    return {"status": overall, "checks": checks}


@app.get("/", tags=["Health"])
async def root():
    return {"name": settings.app_name, "version": settings.app_version, "docs": "/docs"}
