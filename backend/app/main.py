"""FastAPI app entrypoint."""
import asyncio
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .db import close_pool, get_conn
from .routes import (
    chat,
    data_rooms,
    deals,
    dealcloud_sync,
    entities,
    health,
    internal,
    investors,
    orgs,
    sessions,
    slack,
    versions,
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="deal_research_workflow API",
    version="0.0.1",
    description="Backend for the deal-research workflow app. See README.",
    docs_url="/docs" if settings.expose_api_docs else None,
    redoc_url="/redoc" if settings.expose_api_docs else None,
    openapi_url="/openapi.json" if settings.expose_api_docs else None,
)

if settings.origin_list:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

app.include_router(health.router)
app.include_router(sessions.router, prefix="/api/v1")
app.include_router(versions.router, prefix="/api/v1")
app.include_router(orgs.router, prefix="/api/v1")
app.include_router(chat.router, prefix="/api/v1")
app.include_router(entities.router, prefix="/api/v1")
app.include_router(data_rooms.router, prefix="/api/v1")
app.include_router(deals.router, prefix="/api/v1")
app.include_router(dealcloud_sync.router, prefix="/api/v1")
app.include_router(investors.router, prefix="/api/v1")
# Slack endpoints (Todd the Walrus). Mounted at /slack -- signature
# verification is per-route, NOT global, so the /api/v1 routes still
# use X-User-Email auth.
app.include_router(slack.router)
# Service-to-service callbacks from deal_cloud_enhancer. Mounted at
# /internal -- shared-secret auth (X-Internal-Secret), not X-User-Email
# or Slack signing.
app.include_router(internal.router)


def _warmup_blocking() -> None:
    """Cold-start work that the first user request would otherwise pay
    for: open a DB connection (TLS to Neon + pool init), and embed a
    throwaway query (OpenAI client + httpx warm-up + Neon pgvector
    page cache for the org embedding HNSW). Without this, the first
    `find_organizations` call after deploy can spike to 60-90s before
    settling to ~1s warm. Best-effort -- any failure prints and gives
    up so the service stays live.

    Uses print() (not logging) because uvicorn's default logger config
    drops INFO from non-uvicorn loggers; the existing slack/events
    breadcrumbs in this app use print() for the same reason.
    """
    import time as _time
    t0 = _time.time()
    try:
        with get_conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT 1")
            cur.fetchone()
        print(f"[warmup] db pool primed in {(_time.time()-t0)*1000:.0f}ms", flush=True)
    except Exception as e:
        print(f"[warmup] db ping failed: {e}", flush=True)

    t1 = _time.time()
    try:
        from .services.embed import embed_query, EmbedNotConfigured
        try:
            embed_query("warmup")
            print(f"[warmup] openai embed primed in {(_time.time()-t1)*1000:.0f}ms", flush=True)
        except EmbedNotConfigured:
            print("[warmup] openai embed skipped (key not set)", flush=True)
    except Exception as e:
        print(f"[warmup] openai embed failed: {e}", flush=True)


@app.on_event("startup")
async def _on_startup() -> None:
    # Run the blocking warmup in a thread so /healthz responds
    # immediately. The first user request that happens DURING warmup
    # still pays cold-start; only requests after the ~1-2s warmup
    # window benefit. That's fine — Render's healthz gate keeps the
    # service marked "deploying" until it accepts traffic, and the
    # first real Slack/chat request typically arrives seconds later.
    asyncio.get_event_loop().run_in_executor(None, _warmup_blocking)


@app.on_event("shutdown")
def _on_shutdown() -> None:
    close_pool()
