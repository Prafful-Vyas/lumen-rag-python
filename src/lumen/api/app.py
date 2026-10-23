from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from lumen.api.routes import admin_router, auth_router, documents_router, query_router, workspaces_router
from lumen.composition import Container, build_container
from lumen.config import Settings, get_settings
from lumen.utils import get_logger

log = get_logger("lumen.api")


def create_app(settings: Settings | None = None, container: Container | None = None) -> FastAPI:
    """Pass `container` to reuse an already-built one (tests); otherwise one is built on startup."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        owned = container is None
        app.state.container = container or await build_container(settings or get_settings())
        s = app.state.container.settings
        log.info(
            "Lumen API ready; ai=%s vector=%s storage=%s",
            "openai" if s.using_real_ai_providers else "mock",
            s.vector_backend,
            s.storage_backend,
        )
        yield
        if owned:
            await app.state.container.close()

    app = FastAPI(title="Lumen", version="0.1.0", lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

    @app.get("/health")
    async def health() -> dict[str, bool]:
        return {"ok": True}

    for router in (auth_router, documents_router, query_router, workspaces_router, admin_router):
        app.include_router(router)
    return app


app = create_app()
