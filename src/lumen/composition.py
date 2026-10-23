"""Composition root: the one place concrete adapters are chosen from env vars and wired into the
domain services. The API, the worker and the eval harness all build from here, so adapter
selection can never drift between processes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pymongo import AsyncMongoClient
from pymongo.asynchronous.database import AsyncDatabase
from redis.asyncio import Redis

from lumen.adapters.completion import MockCompletionProvider, OpenAICompletionProvider
from lumen.adapters.embedding import MockEmbeddingProvider, OpenAIEmbeddingProvider
from lumen.adapters.reranker import LLMRerankerProvider, MockReranker
from lumen.adapters.storage import LocalDiskStorage, S3Storage
from lumen.adapters.vector_index import AtlasVectorIndex, LocalMongoVectorIndex
from lumen.config import Settings
from lumen.db import connect_mongo, connect_redis, ensure_indexes
from lumen.ingestion.chunker import StructuralChunker
from lumen.ingestion.embed import EmbeddingCache, EmbedService
from lumen.ingestion.extract import ExtractorRegistry
from lumen.ingestion.headers import ContextualHeaderGenerator
from lumen.ingestion.pipeline import IngestionPipeline, InlineJobQueue
from lumen.ports import CompletionProvider, EmbeddingProvider, JobQueue, ObjectStorage, RerankerProvider, VectorIndex
from lumen.query.cache import AnswerCache, MeteringService
from lumen.query.services import GenerationConfig, GenerationService, RetrievalConfig, RetrievalService
from lumen.repositories import ChunkRepository, DocumentRepository, QueryRepository, UserRepository, WorkspaceRepository
from lumen.utils import TokenBucketLimiter, get_logger

log = get_logger("lumen")


def create_embedding_provider(s: Settings) -> EmbeddingProvider:
    if s.using_real_ai_providers:
        return OpenAIEmbeddingProvider(s.openai_api_key, s.embedding_model, s.embedding_dimensions)
    log.warning("OPENAI_API_KEY not set: using MockEmbeddingProvider (deterministic, not semantically meaningful)")
    return MockEmbeddingProvider(s.embedding_dimensions)


def create_completion_provider(s: Settings, model: str | None = None) -> CompletionProvider:
    if s.using_real_ai_providers:
        return OpenAICompletionProvider(s.openai_api_key, model or s.completion_model)
    return MockCompletionProvider()


def create_reranker(s: Settings, completion: CompletionProvider) -> RerankerProvider:
    return LLMRerankerProvider(completion) if s.using_real_ai_providers else MockReranker()


def create_storage(s: Settings) -> ObjectStorage:
    if s.storage_backend == "s3":
        if not s.aws_s3_bucket:
            raise ValueError("STORAGE_BACKEND=s3 requires AWS_S3_BUCKET")
        return S3Storage(
            s.aws_s3_bucket, s.aws_region, s.aws_access_key_id, s.aws_secret_access_key, s.aws_endpoint_url
        )
    return LocalDiskStorage(s.storage_root)


def create_vector_index(s: Settings, db: AsyncDatabase, chunks: ChunkRepository) -> VectorIndex:
    return AtlasVectorIndex(db) if s.vector_backend == "atlas" else LocalMongoVectorIndex(chunks)


@dataclass
class Container:
    settings: Settings
    mongo: AsyncMongoClient
    db: AsyncDatabase
    redis: Redis
    storage: ObjectStorage
    workspaces: WorkspaceRepository
    users: UserRepository
    documents: DocumentRepository
    chunks: ChunkRepository
    queries: QueryRepository
    embedding: EmbeddingProvider
    retrieval: RetrievalService
    generation: GenerationService
    metering: MeteringService
    pipeline: IngestionPipeline
    queue: JobQueue
    _arq_pool: object | None = None

    async def close(self) -> None:
        if self._arq_pool is not None:
            await self._arq_pool.aclose()
        await self.redis.aclose()
        await self.mongo.close()


async def build_container(settings: Settings, queue: Literal["arq", "inline"] = "arq") -> Container:
    mongo, db = connect_mongo(settings)
    await ensure_indexes(db)
    redis = connect_redis(settings)

    workspaces = WorkspaceRepository(db, settings.workspace_default_monthly_cap_usd)
    users = UserRepository(db)
    documents = DocumentRepository(db)
    chunks = ChunkRepository(db)
    queries = QueryRepository(db)

    embedding = create_embedding_provider(settings)
    completion = create_completion_provider(settings)
    header_completion = create_completion_provider(settings, settings.contextual_header_model)
    storage = create_storage(settings)
    embedding_cache = EmbeddingCache(redis)
    metering = MeteringService(redis)

    retrieval = RetrievalService(
        embedding,
        create_vector_index(settings, db, chunks),
        create_reranker(settings, completion),
        embedding_cache,
        RetrievalConfig(
            settings.vector_candidate_limit, settings.lexical_candidate_limit, settings.rrf_k, settings.rerank_top_k
        ),
    )
    generation = GenerationService(
        retrieval=retrieval,
        completion=completion,
        answer_cache=AnswerCache(redis),
        metering=metering,
        chunks=chunks,
        documents=documents,
        workspaces=workspaces,
        queries=queries,
        config=GenerationConfig(
            settings.context_token_budget, settings.refusal_score_threshold, settings.near_duplicate_cosine_threshold
        ),
    )

    arq_pool = None
    if queue == "arq":
        from lumen.ingestion.worker import ArqJobQueue, create_arq_pool

        arq_pool = await create_arq_pool(settings)
        job_queue: JobQueue = ArqJobQueue(arq_pool)
    else:
        job_queue = InlineJobQueue()

    pipeline = IngestionPipeline(
        documents=documents,
        chunks=chunks,
        workspaces=workspaces,
        storage=storage,
        registry=ExtractorRegistry(),
        chunker=StructuralChunker(),
        header_generator=ContextualHeaderGenerator(header_completion, settings.using_real_ai_providers),
        # Outbound embedding budget (200 burst, 50/s), independent of the queue's job concurrency.
        embed_service=EmbedService(embedding, embedding_cache, TokenBucketLimiter(200, 50), chunks),
        embedding_model=embedding.model,
        queue=job_queue,
    )
    if isinstance(job_queue, InlineJobQueue):
        job_queue.pipeline = pipeline

    return Container(
        settings=settings,
        mongo=mongo,
        db=db,
        redis=redis,
        storage=storage,
        workspaces=workspaces,
        users=users,
        documents=documents,
        chunks=chunks,
        queries=queries,
        embedding=embedding,
        retrieval=retrieval,
        generation=generation,
        metering=metering,
        pipeline=pipeline,
        queue=job_queue,
        _arq_pool=arq_pool,
    )
