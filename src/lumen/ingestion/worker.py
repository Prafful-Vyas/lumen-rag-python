"""arq worker process: one arq Worker per stage queue, each with its own concurrency, all running in
one event loop. Run with `python -m lumen.ingestion.worker`."""

from __future__ import annotations

import asyncio
from typing import Any

from arq import Retry, create_pool
from arq.connections import ArqRedis, RedisSettings
from arq.worker import Worker, func

from lumen.config import Settings, get_settings
from lumen.ingestion.pipeline import RETRY_POLICIES, STAGE_CONCURRENCY, STAGES, DocumentMissingError
from lumen.utils import get_logger

log = get_logger("lumen.worker")


def queue_name(stage: str) -> str:
    return f"lumen:ingest:{stage}"


class ArqJobQueue:
    def __init__(self, pool: ArqRedis) -> None:
        self.pool = pool

    async def enqueue(self, stage: str, document_id: str, workspace_id: str) -> None:
        await self.pool.enqueue_job(
            f"ingest_{stage}",
            document_id,
            workspace_id,
            _queue_name=queue_name(stage),
            _job_id=f"{stage}:{document_id}",
        )


async def create_arq_pool(settings: Settings) -> ArqRedis:
    return await create_pool(RedisSettings.from_dsn(settings.redis_url))


async def run_job(ctx: dict[str, Any], stage: str, document_id: str, workspace_id: str) -> None:
    pipeline = ctx["pipeline"]
    try:
        await pipeline.run_stage(stage, document_id, workspace_id)
    except DocumentMissingError:
        log.warning("%s: document %s no longer exists; dropping job", stage, document_id)
    except Exception as exc:
        policy = RETRY_POLICIES[stage]
        attempt = ctx["job_try"]
        if attempt < policy.attempts:
            delay = policy.delay_for(attempt)
            log.warning(
                "%s attempt %s/%s failed for %s, retrying in %ss: %s",
                stage,
                attempt,
                policy.attempts,
                document_id,
                delay,
                exc,
            )
            raise Retry(defer=delay) from exc
        await pipeline.on_exhausted(stage, document_id, exc)
        raise


def _job(stage: str):
    async def handler(ctx: dict[str, Any], document_id: str, workspace_id: str) -> None:
        await run_job(ctx, stage, document_id, workspace_id)

    handler.__name__ = handler.__qualname__ = f"ingest_{stage}"
    return func(handler, name=f"ingest_{stage}", max_tries=RETRY_POLICIES[stage].attempts)


async def main() -> None:
    from lumen.composition import build_container

    settings = get_settings()
    container = await build_container(settings)
    redis_settings = RedisSettings.from_dsn(settings.redis_url)
    workers = [
        Worker(
            functions=[_job(stage)],
            queue_name=queue_name(stage),
            redis_settings=redis_settings,
            max_jobs=STAGE_CONCURRENCY[stage],
            ctx={"pipeline": container.pipeline},
            handle_signals=False,
            keep_result=3600,
        )
        for stage in STAGES
    ]
    log.info(
        "ingestion workers started (extract, chunk, embed, index); ai=%s vector=%s storage=%s",
        "openai" if settings.using_real_ai_providers else "mock",
        settings.vector_backend,
        settings.storage_backend,
    )
    try:
        await asyncio.gather(*(w.main() for w in workers))
    finally:
        await asyncio.gather(*(w.close() for w in workers), return_exceptions=True)
        await container.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
