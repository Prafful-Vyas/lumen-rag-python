from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import time
from functools import lru_cache

import numpy as np
import tiktoken


def get_logger(name: str) -> logging.Logger:
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=os.environ.get("LOG_LEVEL", "INFO"),
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
    return logging.getLogger(name)


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def embedding_cache_key(model: str, text: str) -> str:
    """Same model + same text must always hit."""
    return f"emb:{sha256(f'{model}:{text}')}"


def cosine_similarity(a: list[float] | np.ndarray, b: list[float] | np.ndarray) -> float:
    va, vb = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    n = min(va.size, vb.size)
    if n == 0:
        return 0.0
    va, vb = va[:n], vb[:n]
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    return 0.0 if denom == 0 else float(va @ vb / denom)


@lru_cache(maxsize=1)
def _encoding() -> tiktoken.Encoding:
    return tiktoken.get_encoding("cl100k_base")


def encode(text: str) -> list[int]:
    return _encoding().encode(text, disallowed_special=())


def decode(tokens: list[int]) -> str:
    return _encoding().decode(tokens)


def count_tokens(text: str) -> int:
    return len(encode(text))


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    tokens = encode(text)
    return text if len(tokens) <= max_tokens else decode(tokens[:max_tokens])


class TokenBucketLimiter:
    """In-process token bucket that keeps the embed worker's outbound request rate under the
    provider's rate limit, however many jobs are in flight."""

    def __init__(self, capacity: float, refill_per_second: float) -> None:
        self.capacity = capacity
        self.tokens = capacity
        self.refill_per_second = refill_per_second
        self.last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    def _refill(self) -> None:
        now = time.monotonic()
        self.tokens = min(self.capacity, self.tokens + (now - self.last_refill) * self.refill_per_second)
        self.last_refill = now

    async def acquire(self, count: float = 1) -> None:
        # A request larger than the bucket could never be satisfied; clamp so it waits for a full bucket.
        count = min(count, self.capacity)
        async with self._lock:
            while True:
                self._refill()
                if self.tokens >= count:
                    self.tokens -= count
                    return
                await asyncio.sleep(max((count - self.tokens) / self.refill_per_second, 0.01))
