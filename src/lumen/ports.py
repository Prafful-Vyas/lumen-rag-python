"""Ports (hexagonal architecture). Everything that talks to the outside world sits behind one of
these protocols, and the concrete adapter is chosen by env var in composition.py, so the domain
code never knows whether it is talking to OpenAI or a mock, Atlas or local Mongo, S3 or disk."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Literal, Protocol

from lumen.types import ChunkFilters, RetrievedChunk


@dataclass
class ChatMessage:
    role: Literal["system", "user", "assistant"]
    content: str


@dataclass
class CompletionUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass
class TokenEvent:
    token: str


@dataclass
class DoneEvent:
    usage: CompletionUsage


class EmbeddingProvider(Protocol):
    model: str
    dimensions: int

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Batch-embeds texts in order. Callers own batch size and caching."""

    async def embed_query(self, text: str) -> list[float]:
        """Embeds one query string (cached separately from document embeddings)."""


class CompletionProvider(Protocol):
    model: str

    async def complete(
        self, messages: list[ChatMessage], max_tokens: int | None = None, temperature: float = 0.2
    ) -> tuple[str, CompletionUsage]:
        """Non-streaming call, used for cheap batched tasks such as contextual chunk headers."""

    def stream_complete(
        self, messages: list[ChatMessage], max_tokens: int | None = None, temperature: float = 0.2
    ) -> AsyncIterator[TokenEvent | DoneEvent]:
        """Streaming call for the main answer path: yields TokenEvents, then exactly one DoneEvent."""

    def estimate_cost_usd(self, usage: CompletionUsage) -> float: ...


@dataclass
class RerankCandidate:
    id: str
    text: str


@dataclass
class RerankResult:
    id: str
    score: float


class RerankerProvider(Protocol):
    async def rerank(self, query: str, candidates: list[RerankCandidate], top_k: int) -> list[RerankResult]: ...


class VectorIndex(Protocol):
    async def vector_search(
        self, workspace_id: str, query_vector: list[float], filters: ChunkFilters, limit: int
    ) -> list[RetrievedChunk]: ...

    async def lexical_search(
        self, workspace_id: str, query_text: str, filters: ChunkFilters, limit: int
    ) -> list[RetrievedChunk]: ...


class ObjectStorage(Protocol):
    async def put(self, key: str, data: bytes, content_type: str) -> None: ...

    async def get(self, key: str) -> bytes: ...

    async def delete(self, key: str) -> None: ...


class JobQueue(Protocol):
    """Enqueues the next ingestion stage. ArqJobQueue in production; InlineJobQueue runs the stage
    in-process (tests and the eval harness)."""

    async def enqueue(self, stage: str, document_id: str, workspace_id: str) -> None: ...
