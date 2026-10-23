from __future__ import annotations

import asyncio
import json
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from redis.asyncio import Redis

from lumen.ports import EmbeddingProvider
from lumen.repositories import ChunkRepository
from lumen.utils import TokenBucketLimiter, embedding_cache_key, get_logger

log = get_logger("lumen.embed")

EMBEDDING_TTL_SECONDS = 60 * 60 * 24 * 30
BATCH_SIZE = 96
MAX_ATTEMPTS = 5


class EmbeddingCache:
    """Redis cache keyed by sha256(model + text). Re-uploading a document, or two documents sharing
    boilerplate (every contract has the same governing-law clause), costs nothing to re-embed."""

    def __init__(self, redis: Redis) -> None:
        self.redis = redis

    async def get_many(self, model: str, texts: list[str]) -> list[list[float] | None]:
        if not texts:
            return []
        raw = await self.redis.mget([embedding_cache_key(model, t) for t in texts])
        return [json.loads(v) if v else None for v in raw]

    async def set_many(self, model: str, entries: list[tuple[str, list[float]]]) -> None:
        if not entries:
            return
        async with self.redis.pipeline(transaction=False) as pipe:
            for text, vector in entries:
                pipe.setex(embedding_cache_key(model, text), EMBEDDING_TTL_SECONDS, json.dumps(vector))
            await pipe.execute()


@dataclass
class EmbedOutcome:
    embedded: int
    failed: int
    tokens_processed: int


class EmbedService:
    """Embeds every not-yet-embedded chunk of a document in batches of 96: below ~50 per-request
    overhead dominates, above ~128 a single 429 wastes more retried work than it saves. Cache hits
    skip the provider entirely; only misses spend the rate limiter's budget."""

    def __init__(
        self,
        provider: EmbeddingProvider,
        cache: EmbeddingCache,
        limiter: TokenBucketLimiter,
        chunks: ChunkRepository,
        backoff_base_seconds: float = 1.0,
    ) -> None:
        self.provider = provider
        self.cache = cache
        self.limiter = limiter
        self.chunks = chunks
        self.backoff_base_seconds = backoff_base_seconds

    async def embed_document(
        self, document_id: str, on_progress: Callable[[int, int], Awaitable[None]] | None = None
    ) -> EmbedOutcome:
        pending = await self.chunks.find_pending_embeddings(document_id)
        done = failed = tokens = 0
        for start in range(0, len(pending), BATCH_SIZE):
            batch = pending[start : start + BATCH_SIZE]
            texts = [c["embedText"] for c in batch]
            cached = await self.cache.get_many(self.provider.model, texts)
            misses = [j for j, v in enumerate(cached) if v is None]
            fresh: list[list[float]] = []
            if misses:
                await self.limiter.acquire(len(misses))
                fresh = await self._embed_with_retry([texts[j] for j in misses])

            fresh_by_pos = dict(zip(misses, fresh, strict=False))
            updates: list[tuple[str, list[float]]] = []
            to_cache: list[tuple[str, list[float]]] = []
            for j, chunk in enumerate(batch):
                vector = fresh_by_pos.get(j) if j in fresh_by_pos else cached[j]
                if not vector:
                    failed += 1
                    continue
                updates.append((str(chunk["_id"]), vector))
                if j in fresh_by_pos:
                    to_cache.append((texts[j], vector))
                tokens += chunk["tokenCount"]

            await asyncio.gather(
                self.chunks.bulk_set_embeddings(updates), self.cache.set_many(self.provider.model, to_cache)
            )
            done += len(batch)
            if on_progress:
                await on_progress(done, len(pending))
        return EmbedOutcome(embedded=done - failed, failed=failed, tokens_processed=tokens)

    async def _embed_with_retry(self, texts: list[str]) -> list[list[float]]:
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                return await self.provider.embed(texts)
            except Exception as exc:
                if attempt == MAX_ATTEMPTS:
                    log.error("embedding batch failed after %s attempts; leaving chunks unembedded: %s", attempt, exc)
                    return []
                delay = self.backoff_base_seconds * 2**attempt + random.random() * 0.25
                log.warning("embedding batch failed (attempt %s), retrying in %.1fs: %s", attempt, delay, exc)
                await asyncio.sleep(delay)
        return []
