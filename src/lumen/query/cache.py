from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from redis.asyncio import Redis

from lumen.types import ChunkFilters
from lumen.utils import sha256

ANSWER_CACHE_TTL_SECONDS = 60 * 60
USAGE_KEY_TTL_SECONDS = 70 * 86_400
WARNING_RATIO = 0.8


class AnswerCache:
    """Full-answer cache keyed on workspace + normalized query + filters + corpusVersion. Folding
    corpusVersion into the key is the load-bearing detail: without it, a document uploaded 20
    seconds ago could be silently ignored by a cached answer."""

    def __init__(self, redis: Redis) -> None:
        self.redis = redis

    @staticmethod
    def key(workspace_id: str, normalized: str, filters: ChunkFilters, corpus_version: int) -> str:
        filter_key = json.dumps(filters.cache_key(), sort_keys=True)
        return f"answer:{sha256(f'{workspace_id}:{normalized}:{filter_key}:{corpus_version}')}"

    async def get(
        self, workspace_id: str, normalized: str, filters: ChunkFilters, corpus_version: int
    ) -> dict[str, Any] | None:
        raw = await self.redis.get(self.key(workspace_id, normalized, filters, corpus_version))
        return json.loads(raw) if raw else None

    async def set(
        self, workspace_id: str, normalized: str, filters: ChunkFilters, corpus_version: int, value: dict[str, Any]
    ) -> None:
        payload = {**value, "cachedAt": datetime.now(UTC).isoformat()}
        await self.redis.setex(
            self.key(workspace_id, normalized, filters, corpus_version), ANSWER_CACHE_TTL_SECONDS, json.dumps(payload)
        )


class QuotaExceededError(Exception):
    def __init__(self, spend_usd: float, cap_usd: float) -> None:
        super().__init__(f"Workspace spend ${spend_usd:.4f} exceeds monthly cap ${cap_usd:.2f}")
        self.spend_usd = spend_usd
        self.cap_usd = cap_usd


@dataclass
class MeterResult:
    spend_usd: float
    queries: int
    near_cap: bool


def _month() -> str:
    return datetime.now(UTC).strftime("%Y-%m")


class MeteringService:
    """Redis per-workspace, per-month spend counters. Redis is a fast cache here, not a ledger:
    the `queries` collection stays the billing source of truth and should be reconciled nightly."""

    def __init__(self, redis: Redis) -> None:
        self.redis = redis

    async def meter(self, workspace_id: str, usd: float, cap_usd: float) -> MeterResult:
        key = f"usage:{workspace_id}:{_month()}"
        async with self.redis.pipeline(transaction=True) as pipe:
            pipe.hincrbyfloat(key, "usd", usd)
            pipe.hincrby(key, "queries", 1)
            pipe.expire(key, USAGE_KEY_TTL_SECONDS)
            spend, queries, _ = await pipe.execute()
        spend = float(spend)
        if spend > cap_usd:
            raise QuotaExceededError(spend, cap_usd)
        return MeterResult(spend, int(queries), spend > cap_usd * WARNING_RATIO)

    async def get_usage(self, workspace_id: str, month: str | None = None) -> tuple[float, int]:
        usd, queries = await self.redis.hmget(f"usage:{workspace_id}:{month or _month()}", ["usd", "queries"])
        return float(usd or 0), int(queries or 0)
