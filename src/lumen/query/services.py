from __future__ import annotations

import asyncio
import re
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, replace
from typing import Any

from lumen.ingestion.embed import EmbeddingCache
from lumen.ports import CompletionProvider, DoneEvent, EmbeddingProvider, RerankCandidate, RerankerProvider, VectorIndex
from lumen.query.cache import AnswerCache, MeteringService, QuotaExceededError
from lumen.query.core import (
    REFUSAL_MESSAGE,
    PromptChunk,
    assert_no_tenant_leak,
    build_prompt,
    normalize_query,
    reciprocal_rank_fusion,
    should_refuse,
    validate_citations,
)
from lumen.repositories import ChunkRepository, DocumentRepository, QueryRepository, WorkspaceRepository
from lumen.types import ChunkFilters, FusedChunk, SseEvent, Usage
from lumen.utils import get_logger

log = get_logger("lumen.query")


@dataclass
class RetrievalConfig:
    vector_candidate_limit: int = 40
    lexical_candidate_limit: int = 40
    rrf_k: int = 60
    rerank_top_k: int = 8


@dataclass
class RetrievalResult:
    normalized: str
    fused: list[FusedChunk]
    reranked: list[FusedChunk]
    timings: dict[str, float]


def _ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 2)


class RetrievalService:
    """The read path: normalize -> (embed query || lexical search) -> vector search -> RRF ->
    tenant-leak assertion -> rerank. Provider-agnostic: swapping VECTOR_BACKEND or the reranker
    leaves this class unchanged."""

    def __init__(
        self,
        embedding: EmbeddingProvider,
        vector_index: VectorIndex,
        reranker: RerankerProvider,
        query_cache: EmbeddingCache,
        config: RetrievalConfig,
    ) -> None:
        self.embedding = embedding
        self.vector_index = vector_index
        self.reranker = reranker
        self.query_cache = query_cache
        self.config = config

    async def retrieve(self, workspace_id: str, question: str, filters: ChunkFilters) -> RetrievalResult:
        normalized = normalize_query(question)

        t0 = time.perf_counter()
        query_vector, lexical = await asyncio.gather(
            self._embed_query_cached(normalized),
            self.vector_index.lexical_search(workspace_id, normalized, filters, self.config.lexical_candidate_limit),
        )
        embed_ms = _ms(t0)

        t1 = time.perf_counter()
        vector = await self.vector_index.vector_search(
            workspace_id, query_vector, filters, self.config.vector_candidate_limit
        )
        limit = max(self.config.vector_candidate_limit, self.config.lexical_candidate_limit)
        fused = reciprocal_rank_fusion([vector, lexical], self.config.rrf_k)[:limit]
        assert_no_tenant_leak(workspace_id, fused)
        retrieve_ms = _ms(t1)

        t2 = time.perf_counter()
        results = await self.reranker.rerank(
            normalized, [RerankCandidate(f.chunk_id, f.text) for f in fused], self.config.rerank_top_k
        )
        by_id = {f.chunk_id: f for f in fused}
        reranked = [replace(by_id[r.id], rerank_score=r.score) for r in results if r.id in by_id]
        rerank_ms = _ms(t2)

        return RetrievalResult(
            normalized, fused, reranked, {"embedMs": embed_ms, "retrieveMs": retrieve_ms, "rerankMs": rerank_ms}
        )

    async def _embed_query_cached(self, normalized: str) -> list[float]:
        (cached,) = await self.query_cache.get_many(self.embedding.model, [normalized])
        if cached:
            return cached
        vector = await self.embedding.embed_query(normalized)
        await self.query_cache.set_many(self.embedding.model, [(normalized, vector)])
        return vector


@dataclass
class GenerationConfig:
    context_token_budget: int = 6000
    refusal_score_threshold: float = 0.15
    near_duplicate_cosine_threshold: float = 0.95


class GenerationService:
    """The full question -> answer path: cache check, retrieval, refusal policy, prompt assembly,
    streamed generation, citation validation, metering and the audit record. It yields the same
    SSE events whether the answer came from cache or a fresh generation."""

    def __init__(
        self,
        *,
        retrieval: RetrievalService,
        completion: CompletionProvider,
        answer_cache: AnswerCache,
        metering: MeteringService,
        chunks: ChunkRepository,
        documents: DocumentRepository,
        workspaces: WorkspaceRepository,
        queries: QueryRepository,
        config: GenerationConfig,
    ) -> None:
        self.retrieval = retrieval
        self.completion = completion
        self.answer_cache = answer_cache
        self.metering = metering
        self.chunks = chunks
        self.documents = documents
        self.workspaces = workspaces
        self.queries = queries
        self.config = config

    async def stream_answer(
        self, workspace_id: str, user_id: str, question: str, filters: ChunkFilters
    ) -> AsyncIterator[SseEvent]:
        start = time.perf_counter()
        normalized = normalize_query(question)
        workspace = await self.workspaces.find_by_id(workspace_id)
        if workspace is None:
            yield SseEvent("error", {"message": "Workspace not found"})
            return

        spend, _ = await self.metering.get_usage(workspace_id)
        if spend >= workspace["monthlyCapUsd"]:
            yield SseEvent("error", {"message": "Workspace monthly cost cap reached"})
            return

        cached = await self.answer_cache.get(workspace_id, normalized, filters, workspace["corpusVersion"])
        if cached:
            async for evt in self._emit_cached(workspace_id, user_id, question, normalized, cached, start):
                yield evt
            return

        async for evt in self._generate_fresh(workspace, user_id, question, normalized, filters, start):
            yield evt

    async def _emit_cached(
        self, workspace_id: str, user_id: str, question: str, normalized: str, cached: dict[str, Any], start: float
    ) -> AsyncIterator[SseEvent]:
        yield SseEvent("sources", {"sources": cached["sources"], "refused": cached["refused"], "cached": True})
        for word in re.split(r"(?<= )", cached["answer"]):
            if word:
                yield SseEvent("token", {"t": word})
        query_id = None
        try:
            query_id = await self.queries.record(
                workspace_id=workspace_id,
                user_id=user_id,
                question=question,
                normalized=normalized,
                retrieved=[],
                used_chunk_ids=[s["chunkId"] for s in cached["sources"]],
                answer=cached["answer"],
                refused=cached["refused"],
                citations=[{"marker": s["marker"], "chunkId": s["chunkId"]} for s in cached["sources"]],
                invalid_citations=cached["invalidCitationCount"],
                timings={"embedMs": 0, "retrieveMs": 0, "rerankMs": 0, "ttftMs": _ms(start), "totalMs": _ms(start)},
                usage=cached["usage"],
            )
        except Exception as exc:  # analytics only; never fail a cached answer over it
            log.error("failed to record cached query: %s", exc)
        yield SseEvent(
            "done",
            {
                "usage": cached["usage"],
                "latencyMs": _ms(start),
                "invalidCitations": cached["invalidCitationCount"],
                "queryId": query_id,
                "cached": True,
            },
        )

    async def _generate_fresh(
        self,
        workspace: dict[str, Any],
        user_id: str,
        question: str,
        normalized: str,
        filters: ChunkFilters,
        start: float,
    ) -> AsyncIterator[SseEvent]:
        workspace_id = str(workspace["_id"])
        corpus_version = workspace["corpusVersion"]
        result = await self.retrieval.retrieve(workspace_id, question, filters)
        top = result.reranked[0] if result.reranked else None

        if should_refuse(top, self.config.refusal_score_threshold):
            zero = Usage().to_dict()
            yield SseEvent("sources", {"sources": [], "refused": True, "cached": False})
            yield SseEvent("token", {"t": REFUSAL_MESSAGE})
            query_id = await self.queries.record(
                workspace_id=workspace_id,
                user_id=user_id,
                question=question,
                normalized=normalized,
                retrieved=result.fused,
                used_chunk_ids=[],
                answer=REFUSAL_MESSAGE,
                refused=True,
                citations=[],
                invalid_citations=0,
                timings={**result.timings, "ttftMs": _ms(start), "totalMs": _ms(start)},
                usage=zero,
            )
            await self.answer_cache.set(
                workspace_id,
                normalized,
                filters,
                corpus_version,
                {"answer": REFUSAL_MESSAGE, "sources": [], "usage": zero, "refused": True, "invalidCitationCount": 0},
            )
            yield SseEvent(
                "done",
                {"usage": zero, "latencyMs": _ms(start), "invalidCitations": 0, "queryId": query_id, "cached": False},
            )
            return

        chunk_docs = await self.chunks.find_by_ids(workspace_id, [c.chunk_id for c in result.reranked])
        embedding_by_id = {str(c["_id"]): c.get("embedding") or [] for c in chunk_docs}
        prompt_chunks = [PromptChunk(c, embedding_by_id.get(c.chunk_id, [])) for c in result.reranked]
        titles = await self.documents.find_titles_by_ids(workspace_id, [c.document_id for c in result.reranked])
        messages, sources = build_prompt(
            question,
            titles,
            prompt_chunks,
            self.config.context_token_budget,
            self.config.near_duplicate_cosine_threshold,
        )

        # Sources go out before the first token: the user sees which documents are being consulted
        # well before prose starts streaming.
        source_dicts = [s.to_dict() for s in sources]
        yield SseEvent("sources", {"sources": source_dicts, "refused": False, "cached": False})

        ttft_ms: float | None = None
        full_text = ""
        usage = None
        async for evt in self.completion.stream_complete(messages, max_tokens=1000):
            if isinstance(evt, DoneEvent):
                usage = evt.usage
                continue
            if ttft_ms is None:
                ttft_ms = _ms(start)
            full_text += evt.token
            yield SseEvent("token", {"t": evt.token})

        check = validate_citations(full_text, sources)
        cost = Usage(
            usage.prompt_tokens if usage else 0,
            usage.completion_tokens if usage else 0,
            self.completion.estimate_cost_usd(usage) if usage else 0.0,
        ).to_dict()

        query_id = await self.queries.record(
            workspace_id=workspace_id,
            user_id=user_id,
            question=question,
            normalized=normalized,
            retrieved=result.fused,
            used_chunk_ids=[str(c["chunkId"]) for c in check.citations],
            answer=check.cleaned_text,
            refused=False,
            citations=check.citations,
            invalid_citations=check.invalid_count,
            timings={**result.timings, "ttftMs": ttft_ms or _ms(start), "totalMs": _ms(start)},
            usage=cost,
        )
        await self.answer_cache.set(
            workspace_id,
            normalized,
            filters,
            corpus_version,
            {
                "answer": check.cleaned_text,
                "sources": source_dicts,
                "usage": cost,
                "refused": False,
                "invalidCitationCount": check.invalid_count,
            },
        )
        try:
            await self.metering.meter(workspace_id, cost["usd"], workspace["monthlyCapUsd"])
        except QuotaExceededError as exc:
            # The answer already streamed in full; the cap blocks the *next* request rather than
            # discarding work the user has already received.
            yield SseEvent("error", {"message": f"Workspace monthly cost cap reached: {exc}"})
            return
        yield SseEvent(
            "done",
            {
                "usage": cost,
                "latencyMs": _ms(start),
                "invalidCitations": check.invalid_count,
                "queryId": query_id,
                "cached": False,
            },
        )
