from __future__ import annotations

from typing import Any

import numpy as np
from pymongo.asynchronous.database import AsyncDatabase

from lumen.repositories import ChunkRepository, chunk_filter_query, oid
from lumen.types import BoundingBox, ChunkFilters, RetrievedChunk


def _to_retrieved(c: dict[str, Any], score: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=str(c["_id"]),
        document_id=str(c["documentId"]),
        workspace_id=str(c["workspaceId"]),
        text=c["text"],
        page_number=c["pageNumber"],
        section=c.get("section", ""),
        score=float(score),
        bbox=BoundingBox(**c["bbox"]) if c.get("bbox") else None,
    )


class LocalMongoVectorIndex:
    """Default VECTOR_BACKEND. Brute-force cosine (one numpy matrix-vector product) over every
    live, embedded chunk in the workspace, plus a Mongo $text index for lexical search. It is
    O(chunks in workspace) per query and has none of HNSW's filtered traversal, but it needs only
    a plain MongoDB. Set VECTOR_BACKEND=atlas for a real cluster; nothing above this class changes."""

    def __init__(self, chunks: ChunkRepository) -> None:
        self.chunks = chunks

    async def vector_search(
        self, workspace_id: str, query_vector: list[float], filters: ChunkFilters, limit: int
    ) -> list[RetrievedChunk]:
        candidates = [
            c for c in await self.chunks.find_live_with_embeddings(workspace_id, filters) if c.get("embedding")
        ]
        if not candidates:
            return []
        matrix = np.asarray([c["embedding"] for c in candidates], dtype=np.float32)
        q = np.asarray(query_vector, dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1) * (np.linalg.norm(q) or 1.0)
        scores = (matrix @ q) / np.where(norms == 0, 1.0, norms)
        top = np.argsort(-scores)[:limit]
        return [_to_retrieved(candidates[i], scores[i]) for i in top]

    async def lexical_search(
        self, workspace_id: str, query_text: str, filters: ChunkFilters, limit: int
    ) -> list[RetrievedChunk]:
        results = await self.chunks.text_search(workspace_id, query_text, filters, limit)
        return [_to_retrieved(c, c.get("score", 0.0)) for c in results]


class AtlasVectorIndex:
    """Requires a MongoDB Atlas cluster with the `chunks_vector` ($vectorSearch) and
    `chunks_lexical` ($search) indexes described in the README. numCandidates = limit * 20 is
    where recall plateaus; widening further only costs latency."""

    PROJECTION = {"text": 1, "documentId": 1, "workspaceId": 1, "pageNumber": 1, "bbox": 1, "section": 1}

    def __init__(self, db: AsyncDatabase) -> None:
        self.col = db.chunks

    async def vector_search(
        self, workspace_id: str, query_vector: list[float], filters: ChunkFilters, limit: int
    ) -> list[RetrievedChunk]:
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "chunks_vector",
                    "path": "embedding",
                    "queryVector": query_vector,
                    "numCandidates": limit * 20,
                    "limit": limit,
                    "filter": chunk_filter_query(workspace_id, filters),
                }
            },
            {"$project": {**self.PROJECTION, "score": {"$meta": "vectorSearchScore"}}},
        ]
        cursor = await self.col.aggregate(pipeline)
        return [_to_retrieved(r, r["score"]) async for r in cursor]

    async def lexical_search(
        self, workspace_id: str, query_text: str, filters: ChunkFilters, limit: int
    ) -> list[RetrievedChunk]:
        must_filter: list[dict[str, Any]] = [
            {"equals": {"path": "workspaceId", "value": oid(workspace_id)}},
            {"equals": {"path": "live", "value": True}},
        ]
        if filters.document_ids:
            must_filter.append({"in": {"path": "documentId", "value": [oid(d) for d in filters.document_ids]}})
        pipeline = [
            {
                "$search": {
                    "index": "chunks_lexical",
                    "compound": {"must": [{"text": {"query": query_text, "path": "text"}}], "filter": must_filter},
                }
            },
            {"$limit": limit},
            {"$project": {**self.PROJECTION, "score": {"$meta": "searchScore"}}},
        ]
        cursor = await self.col.aggregate(pipeline)
        return [_to_retrieved(r, r["score"]) async for r in cursor]
