import os
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api import routes_sessions, routes_chat, routes_traces
from app.api.errors import HelixError, helix_error_handler
from app.db.session import init_db
from app.obs.logging import configure_logging
from app.settings import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    # Ensure Google SDK finds the key
    if settings.google_api_key:
        os.environ["GOOGLE_API_KEY"] = settings.google_api_key
    await init_db()
    yield


app = FastAPI(title="Helix SROP", version="0.1.0", lifespan=lifespan)

app.include_router(routes_sessions.router, prefix="/v1")
app.include_router(routes_chat.router, prefix="/v1")
app.include_router(routes_traces.router, prefix="/v1")


@app.middleware("http")
async def clear_structlog_context(request: Request, call_next):
    """Clear per-request structlog context vars so they don't leak between requests."""
    structlog.contextvars.clear_contextvars()
    return await call_next(request)


@app.get("/healthz")
async def healthz() -> dict:
    return {"status": "ok"}


app.add_exception_handler(HelixError, helix_error_handler)  # type: ignore[arg-type]
