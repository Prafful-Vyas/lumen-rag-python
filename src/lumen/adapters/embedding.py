from __future__ import annotations

import hashlib

import numpy as np
from openai import AsyncOpenAI


class OpenAIEmbeddingProvider:
    """text-embedding-3-large truncated to `dimensions` with OpenAI's native Matryoshka support
    (the API truncates and renormalizes server-side). 1024 dims is the knee of recall vs memory."""

    def __init__(self, api_key: str, model: str, dimensions: int) -> None:
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model
        self.dimensions = dimensions

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        res = await self.client.embeddings.create(model=self.model, input=texts, dimensions=self.dimensions)
        return [d.embedding for d in sorted(res.data, key=lambda d: d.index)]

    async def embed_query(self, text: str) -> list[float]:
        vectors = await self.embed([text])
        if not vectors:
            raise RuntimeError("OpenAI embeddings API returned no result for query")
        return vectors[0]


class MockEmbeddingProvider:
    """Deterministic, zero-cost, offline embeddings for local dev and CI. Each text hashes to a
    fixed unit vector, so the same text always gets the same vector and unrelated texts land
    near-orthogonal. That is enough to exercise the whole pipeline; it is not semantically meaningful."""

    model = "mock-embedding"

    def __init__(self, dimensions: int) -> None:
        self.dimensions = dimensions

    def _vector_for(self, text: str) -> list[float]:
        seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "little")
        vec = np.random.default_rng(seed).random(self.dimensions) - 0.5
        norm = float(np.linalg.norm(vec)) or 1.0
        return (vec / norm).tolist()

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vector_for(t) for t in texts]

    async def embed_query(self, text: str) -> list[float]:
        return self._vector_for(text)
