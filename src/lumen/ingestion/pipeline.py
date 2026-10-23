"""Four-stage ingestion: extract -> chunk -> embed -> index. Each stage is its own job with its own
retry policy, because each fails differently: a transient 429 from the embedding provider should
retry the embed step, not force re-extraction and re-chunking of a 300-page PDF."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from lumen.adapters.pricing import estimate_embedding_cost_usd
from lumen.ingestion.chunker import StructuralChunker
from lumen.ingestion.embed import EmbedService
from lumen.ingestion.extract import ExtractorRegistry
from lumen.ingestion.headers import ContextualHeaderGenerator
from lumen.ports import JobQueue, ObjectStorage
from lumen.repositories import ChunkRepository, DocumentRepository, WorkspaceRepository
from lumen.utils import get_logger

log = get_logger("lumen.pipeline")

Stage = Literal["extract", "chunk", "embed", "index"]
STAGES: tuple[Stage, ...] = ("extract", "chunk", "embed", "index")
NEXT_STAGE: dict[str, Stage | None] = {"extract": "chunk", "chunk": "embed", "embed": "index", "index": None}

MAX_CHUNKS_PER_DOCUMENT = 5000
MAX_FAILED_CHUNK_RATIO = 0.05


@dataclass(frozen=True)
class RetryPolicy:
    attempts: int
    backoff: Literal["fixed", "exponential"]
    delay_seconds: float

    def delay_for(self, attempt: int) -> float:
        """Delay before retrying after `attempt` (1-based) failed."""
        return self.delay_seconds if self.backoff == "fixed" else self.delay_seconds * 2 ** (attempt - 1)


RETRY_POLICIES: dict[str, RetryPolicy] = {
    # Corrupt files fail fast: retrying extraction of a genuinely broken PDF just burns time twice.
    "extract": RetryPolicy(2, "fixed", 5),
    # Pure CPU plus one batched LLM call, so a retry is cheap and always safe.
    "chunk": RetryPolicy(3, "exponential", 1),
    # Bound by the provider's rate limit, so back off harder.
    "embed": RetryPolicy(5, "exponential", 4),
    "index": RetryPolicy(5, "exponential", 2),
}

# Embed stays at 2: the provider's rate limit is the real constraint, enforced by the shared token bucket.
STAGE_CONCURRENCY: dict[str, int] = {"extract": 4, "chunk": 8, "embed": 2, "index": 4}


class DocumentMissingError(Exception):
    pass


class IngestionPipeline:
    def __init__(
        self,
        *,
        documents: DocumentRepository,
        chunks: ChunkRepository,
        workspaces: WorkspaceRepository,
        storage: ObjectStorage,
        registry: ExtractorRegistry,
        chunker: StructuralChunker,
        header_generator: ContextualHeaderGenerator,
        embed_service: EmbedService,
        embedding_model: str,
        queue: JobQueue,
    ) -> None:
        self.documents = documents
        self.chunks = chunks
        self.workspaces = workspaces
        self.storage = storage
        self.registry = registry
        self.chunker = chunker
        self.header_generator = header_generator
        self.embed_service = embed_service
        self.embedding_model = embedding_model
        self.queue = queue

    async def run_stage(self, stage: str, document_id: str, workspace_id: str) -> None:
        await getattr(self, stage)(document_id, workspace_id)
        if (nxt := NEXT_STAGE[stage]) is not None:
            await self.queue.enqueue(nxt, document_id, workspace_id)

    async def on_exhausted(self, stage: str, document_id: str, exc: BaseException) -> None:
        """Called only once a stage has used up its retry policy, so an in-flight retry shows as
        still processing in the UI instead of flashing 'failed' between attempts."""
        log.error("%s stage failed permanently for %s: %s", stage, document_id, exc)
        await self.documents.mark_failed(document_id, stage, str(exc))

    async def _require(self, document_id: str) -> dict:
        doc = await self.documents.find_by_id_unscoped(document_id)
        if doc is None:
            raise DocumentMissingError(f"Document {document_id} not found")
        return doc

    async def extract(self, document_id: str, workspace_id: str) -> None:
        doc = await self._require(document_id)
        await self.documents.update_status(document_id, "extracting", 5)
        data = await self.storage.get(doc["storageKey"])
        result = self.registry.extract(data, doc["mimeType"])
        await self.documents.set_extraction_result(document_id, outline=result.outline, pages=result.pages)
        await self.documents.update_progress(document_id, 20)
        log.info("extracted %s: %s pages", document_id, len(result.pages))

    async def chunk(self, document_id: str, workspace_id: str) -> None:
        doc = await self._require(document_id)
        await self.documents.update_status(document_id, "chunking", 25)
        candidates = self.chunker.chunk_document(await self.documents.get_extracted_pages(document_id))
        if len(candidates) > MAX_CHUNKS_PER_DOCUMENT:
            # Cap very long documents rather than let one dominate its workspace's index.
            log.warning("%s has %s chunks; truncating to %s", document_id, len(candidates), MAX_CHUNKS_PER_DOCUMENT)
            candidates = candidates[:MAX_CHUNKS_PER_DOCUMENT]
        headers = await self.header_generator.generate(doc["title"], doc.get("outline", ""), candidates)

        await self.chunks.delete_by_document(document_id)  # a retried chunk stage starts clean
        inserted = await self.chunks.insert_many(
            [
                {
                    "workspaceId": workspace_id,
                    "documentId": document_id,
                    "folderId": doc.get("folderId"),
                    "text": c.text,
                    "embedText": f"[{header}]\n{c.text}",
                    "chunkIndex": i,
                    "pageNumber": c.page_number,
                    "tokenCount": c.token_count,
                    "section": c.section,
                    "uploadedAt": doc["createdAt"],
                }
                for i, (c, header) in enumerate(zip(candidates, headers, strict=True))
            ]
        )
        await self.documents.set_chunk_count(document_id, inserted)
        await self.documents.clear_extracted_pages(document_id)
        await self.documents.update_progress(document_id, 40)
        log.info("chunked %s: %s chunks", document_id, inserted)

    async def embed(self, document_id: str, workspace_id: str) -> None:
        await self.documents.update_status(document_id, "embedding", 40)

        async def progress(done: int, total: int) -> None:
            await self.documents.update_progress(document_id, 40 + round(done / max(total, 1) * 50))

        outcome = await self.embed_service.embed_document(document_id, progress)
        await self.documents.increment_ingest_cost(
            document_id,
            embed_tokens=outcome.tokens_processed,
            usd=estimate_embedding_cost_usd(self.embedding_model, outcome.tokens_processed),
        )
        log.info("embedded %s: %s ok, %s failed", document_id, outcome.embedded, outcome.failed)

    async def index(self, document_id: str, workspace_id: str) -> None:
        """The atomic visibility flip. A document is published whole or fails loudly; above a 5%
        embed-failure rate it is not published at all, because silent partial ingestion is the
        worst possible failure for a search product."""
        missing = await self.chunks.count_missing_embeddings(document_id)
        total = await self.chunks.count_by_document(document_id)
        if missing:
            ratio = missing / total if total else 1.0
            if ratio > MAX_FAILED_CHUNK_RATIO:
                raise RuntimeError(f"{missing}/{total} chunks failed to embed ({ratio:.1%}); refusing to publish")
            log.warning("publishing %s with %s/%s unembedded chunks", document_id, missing, total)
        await self.chunks.publish_live(document_id)
        await self.documents.mark_ready(document_id, missing)
        await self.workspaces.bump_corpus_version(workspace_id)
        log.info("%s is live and searchable", document_id)


class InlineJobQueue:
    """Runs each stage in-process as soon as it is enqueued, applying the same retry policies
    without the waits. Used by tests and the eval harness; production uses ArqJobQueue."""

    def __init__(self) -> None:
        self.pipeline: IngestionPipeline | None = None

    async def enqueue(self, stage: str, document_id: str, workspace_id: str) -> None:
        assert self.pipeline is not None, "InlineJobQueue.pipeline must be set before enqueueing"
        policy = RETRY_POLICIES[stage]
        for attempt in range(1, policy.attempts + 1):
            try:
                await self.pipeline.run_stage(stage, document_id, workspace_id)
                return
            except DocumentMissingError:
                return
            except Exception as exc:
                if attempt == policy.attempts:
                    await self.pipeline.on_exhausted(stage, document_id, exc)
                    return
