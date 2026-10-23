"""Data access. Every method that reads tenant data takes workspace_id and filters on it, so a
route handler cannot express a cross-tenant read. The only unscoped reads are the worker's
lookups by document id and login's lookup by email, and they are named as such."""

from __future__ import annotations

import re
import secrets
from datetime import UTC, datetime
from typing import Any

from bson import ObjectId
from pymongo import DESCENDING, ReturnDocument, UpdateOne
from pymongo.asynchronous.database import AsyncDatabase

from lumen.types import ChunkFilters, ExtractedPage, FusedChunk


def oid(value: str | ObjectId) -> ObjectId:
    return value if isinstance(value, ObjectId) else ObjectId(value)


def now() -> datetime:
    return datetime.now(UTC)


def _slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower().strip()).strip("-")


def chunk_filter_query(workspace_id: str, filters: ChunkFilters, **extra: Any) -> dict[str, Any]:
    query: dict[str, Any] = {"workspaceId": oid(workspace_id), "live": True, **extra}
    if filters.document_ids:
        query["documentId"] = {"$in": [oid(d) for d in filters.document_ids]}
    if filters.folder_id:
        query["folderId"] = oid(filters.folder_id)
    if filters.after:
        query["uploadedAt"] = {"$gte": filters.after}
    return query


class WorkspaceRepository:
    def __init__(self, db: AsyncDatabase, default_monthly_cap_usd: float) -> None:
        self.col = db.workspaces
        self.default_cap = default_monthly_cap_usd

    async def create(self, name: str) -> dict[str, Any]:
        base = _slugify(name) or "workspace"
        slug, suffix = base, 0
        while await self.col.find_one({"slug": slug}, {"_id": 1}):
            suffix += 1
            slug = f"{base}-{suffix}"
        doc = {
            "name": name.strip(),
            "slug": slug,
            "monthlyCapUsd": self.default_cap,
            # Bumped on every document add/delete and folded into the answer-cache key, so a cached
            # answer can never ignore a document that was just uploaded or removed.
            "corpusVersion": 0,
            "inviteCode": secrets.token_urlsafe(9),
            "createdAt": now(),
        }
        res = await self.col.insert_one(doc)
        return {**doc, "_id": res.inserted_id}

    async def find_by_id(self, workspace_id: str) -> dict[str, Any] | None:
        return await self.col.find_one({"_id": oid(workspace_id)})

    async def find_by_invite_code(self, invite_code: str) -> dict[str, Any] | None:
        return await self.col.find_one({"inviteCode": invite_code})

    async def find_by_slug(self, slug: str) -> dict[str, Any] | None:
        return await self.col.find_one({"slug": slug})

    async def bump_corpus_version(self, workspace_id: str) -> int:
        doc = await self.col.find_one_and_update(
            {"_id": oid(workspace_id)}, {"$inc": {"corpusVersion": 1}}, return_document=ReturnDocument.AFTER
        )
        return doc["corpusVersion"] if doc else 0


class UserRepository:
    def __init__(self, db: AsyncDatabase) -> None:
        self.col = db.users

    async def find_by_email(self, email: str) -> dict[str, Any] | None:
        """Unscoped by design: login looks a user up before their workspace is known."""
        return await self.col.find_one({"email": email.lower().strip()})

    async def find_by_id(self, user_id: str) -> dict[str, Any] | None:
        return await self.col.find_one({"_id": oid(user_id)})

    async def create(self, email: str, password_hash: str, workspace_id: str, role: str) -> dict[str, Any]:
        doc = {
            "email": email.lower().strip(),
            "passwordHash": password_hash,
            "workspaceId": oid(workspace_id),
            "role": role,
            "createdAt": now(),
        }
        res = await self.col.insert_one(doc)
        return {**doc, "_id": res.inserted_id}

    async def list_by_workspace(self, workspace_id: str) -> list[dict[str, Any]]:
        return await self.col.find({"workspaceId": oid(workspace_id)}).to_list()


class DocumentRepository:
    def __init__(self, db: AsyncDatabase) -> None:
        self.col = db.documents

    async def create(
        self,
        *,
        document_id: ObjectId,
        workspace_id: str,
        title: str,
        filename: str,
        mime_type: str,
        storage_key: str,
        size_bytes: int,
        uploaded_by: str,
        folder_id: str | None = None,
    ) -> dict[str, Any]:
        doc = {
            "_id": document_id,
            "workspaceId": oid(workspace_id),
            "title": title,
            "filename": filename,
            "mimeType": mime_type,
            "storageKey": storage_key,
            "sizeBytes": size_bytes,
            "pageCount": 0,
            "uploadedBy": oid(uploaded_by),
            "folderId": oid(folder_id) if folder_id else None,
            "status": "queued",
            "progress": 0,
            "error": None,
            "chunkCount": 0,
            "failedChunks": 0,
            "ocrPages": [],
            "outline": "",
            "ingestCost": {"embedTokens": 0, "contextTokens": 0, "usd": 0.0},
            "readyAt": None,
            "deletedAt": None,
            "createdAt": now(),
        }
        await self.col.insert_one(doc)
        return doc

    async def find_by_id(self, workspace_id: str, document_id: str) -> dict[str, Any] | None:
        if not ObjectId.is_valid(document_id):
            return None
        return await self.col.find_one(
            {"_id": oid(document_id), "workspaceId": oid(workspace_id), "deletedAt": None}, {"extractedPages": 0}
        )

    async def find_by_id_unscoped(self, document_id: str) -> dict[str, Any] | None:
        """For the worker, which operates by document id with no request context."""
        return await self.col.find_one({"_id": oid(document_id)}, {"extractedPages": 0})

    async def list(
        self,
        workspace_id: str,
        *,
        folder_id: str | None = None,
        uploaded_by: str | None = None,
        after: datetime | None = None,
        before: datetime | None = None,
    ) -> list[dict[str, Any]]:
        query: dict[str, Any] = {"workspaceId": oid(workspace_id), "deletedAt": None}
        if folder_id:
            query["folderId"] = oid(folder_id)
        if uploaded_by:
            query["uploadedBy"] = oid(uploaded_by)
        if after or before:
            query["createdAt"] = {**({"$gte": after} if after else {}), **({"$lte": before} if before else {})}
        return await self.col.find(query, {"extractedPages": 0}).sort("createdAt", DESCENDING).to_list()

    async def update_status(self, document_id: str, status: str, progress: int | None = None) -> None:
        fields: dict[str, Any] = {"status": status}
        if progress is not None:
            fields["progress"] = progress
        await self.col.update_one({"_id": oid(document_id)}, {"$set": fields})

    async def update_progress(self, document_id: str, progress: int) -> None:
        await self.col.update_one({"_id": oid(document_id)}, {"$set": {"progress": progress}})

    async def mark_failed(self, document_id: str, stage: str, message: str) -> None:
        await self.col.update_one(
            {"_id": oid(document_id)}, {"$set": {"status": "failed", "error": {"stage": stage, "message": message}}}
        )

    async def mark_ready(self, document_id: str, failed_chunks: int) -> None:
        await self.col.update_one(
            {"_id": oid(document_id)},
            {"$set": {"status": "ready", "progress": 100, "readyAt": now(), "failedChunks": failed_chunks}},
        )

    async def set_extraction_result(self, document_id: str, *, outline: str, pages: list[ExtractedPage]) -> None:
        # extractedPages is a checkpoint between the extract and chunk stages, so a chunk-stage retry
        # never re-extracts a 300-page PDF. It is cleared once chunking has consumed it.
        await self.col.update_one(
            {"_id": oid(document_id)},
            {
                "$set": {
                    "pageCount": len(pages),
                    "outline": outline,
                    "ocrPages": [p.page_number for p in pages if p.ocr],
                    "extractedPages": [{"pageNumber": p.page_number, "text": p.text, "ocr": p.ocr} for p in pages],
                }
            },
        )

    async def get_extracted_pages(self, document_id: str) -> list[ExtractedPage]:
        doc = await self.col.find_one({"_id": oid(document_id)}, {"extractedPages": 1})
        return [
            ExtractedPage(page_number=p["pageNumber"], text=p["text"], ocr=p["ocr"])
            for p in (doc or {}).get("extractedPages") or []
        ]

    async def clear_extracted_pages(self, document_id: str) -> None:
        await self.col.update_one({"_id": oid(document_id)}, {"$unset": {"extractedPages": 1}})

    async def set_chunk_count(self, document_id: str, chunk_count: int) -> None:
        await self.col.update_one({"_id": oid(document_id)}, {"$set": {"chunkCount": chunk_count}})

    async def increment_ingest_cost(
        self, document_id: str, *, embed_tokens: int = 0, context_tokens: int = 0, usd: float = 0.0
    ) -> None:
        await self.col.update_one(
            {"_id": oid(document_id)},
            {
                "$inc": {
                    "ingestCost.embedTokens": embed_tokens,
                    "ingestCost.contextTokens": context_tokens,
                    "ingestCost.usd": usd,
                }
            },
        )

    async def soft_delete(self, workspace_id: str, document_id: str) -> None:
        await self.col.update_one(
            {"_id": oid(document_id), "workspaceId": oid(workspace_id)}, {"$set": {"deletedAt": now()}}
        )

    async def find_titles_by_ids(self, workspace_id: str, document_ids: list[str]) -> dict[str, str]:
        if not document_ids:
            return {}
        cursor = self.col.find(
            {"_id": {"$in": [oid(d) for d in set(document_ids)]}, "workspaceId": oid(workspace_id)}, {"title": 1}
        )
        return {str(d["_id"]): d["title"] async for d in cursor}


class ChunkRepository:
    def __init__(self, db: AsyncDatabase) -> None:
        self.col = db.chunks

    async def insert_many(self, chunks: list[dict[str, Any]]) -> int:
        if not chunks:
            return 0
        res = await self.col.insert_many(
            [
                {
                    "workspaceId": oid(c["workspaceId"]),
                    "documentId": oid(c["documentId"]),
                    "folderId": oid(c["folderId"]) if c.get("folderId") else None,
                    "text": c["text"],
                    "embedText": c["embedText"],
                    "embedding": None,
                    "chunkIndex": c["chunkIndex"],
                    "pageNumber": c["pageNumber"],
                    "bbox": None,
                    "tokenCount": c["tokenCount"],
                    "section": c["section"],
                    "live": False,
                    "uploadedAt": c["uploadedAt"],
                    "createdAt": now(),
                }
                for c in chunks
            ],
            ordered=True,
        )
        return len(res.inserted_ids)

    async def delete_by_document(self, document_id: str) -> None:
        """Lets a chunk-stage retry start clean instead of inserting duplicates."""
        await self.col.delete_many({"documentId": oid(document_id)})

    async def find_pending_embeddings(self, document_id: str) -> list[dict[str, Any]]:
        return (
            await self.col.find({"documentId": oid(document_id), "embedding": None}, {"embedText": 1, "tokenCount": 1})
            .sort("chunkIndex", 1)
            .to_list()
        )

    async def bulk_set_embeddings(self, ops: list[tuple[str, list[float]]]) -> None:
        if ops:
            await self.col.bulk_write(
                [UpdateOne({"_id": oid(cid)}, {"$set": {"embedding": vec}}) for cid, vec in ops], ordered=False
            )

    async def count_missing_embeddings(self, document_id: str) -> int:
        return await self.col.count_documents({"documentId": oid(document_id), "embedding": None})

    async def count_by_document(self, document_id: str) -> int:
        return await self.col.count_documents({"documentId": oid(document_id)})

    async def publish_live(self, document_id: str) -> None:
        """Atomic visibility flip: chunks become searchable only once the whole document is embedded."""
        await self.col.update_many(
            {"documentId": oid(document_id), "embedding": {"$ne": None}}, {"$set": {"live": True}}
        )

    async def delist_by_document(self, document_id: str) -> None:
        """Takes effect on the very next query because `live` is filtered on every retrieval path."""
        await self.col.update_many({"documentId": oid(document_id)}, {"$set": {"live": False}})

    async def find_by_ids(self, workspace_id: str, chunk_ids: list[str]) -> list[dict[str, Any]]:
        if not chunk_ids:
            return []
        return await self.col.find(
            {"_id": {"$in": [oid(c) for c in chunk_ids]}, "workspaceId": oid(workspace_id)}
        ).to_list()

    async def find_live_with_embeddings(self, workspace_id: str, filters: ChunkFilters) -> list[dict[str, Any]]:
        """Full scan of a workspace's live, embedded chunks for brute-force cosine. Fine up to a few
        hundred thousand chunks per workspace; past that, switch VECTOR_BACKEND=atlas."""
        return await self.col.find(
            chunk_filter_query(workspace_id, filters, embedding={"$ne": None}),
            {"embedText": 0},
        ).to_list()

    async def text_search(
        self, workspace_id: str, query_text: str, filters: ChunkFilters, limit: int
    ) -> list[dict[str, Any]]:
        return (
            await self.col.find(
                chunk_filter_query(workspace_id, filters, **{"$text": {"$search": query_text}}),
                {"score": {"$meta": "textScore"}, "embedding": 0, "embedText": 0},
            )
            .sort([("score", {"$meta": "textScore"})])
            .limit(limit)
            .to_list()
        )


class QueryRepository:
    """The audit trail behind the eval harness and thumbs-down triage."""

    def __init__(self, db: AsyncDatabase) -> None:
        self.col = db.queries

    async def record(
        self,
        *,
        workspace_id: str,
        user_id: str,
        question: str,
        normalized: str,
        retrieved: list[FusedChunk],
        used_chunk_ids: list[str],
        answer: str,
        refused: bool,
        citations: list[dict[str, Any]],
        invalid_citations: int,
        timings: dict[str, float],
        usage: dict[str, Any],
    ) -> str:
        res = await self.col.insert_one(
            {
                "workspaceId": oid(workspace_id),
                "userId": oid(user_id),
                "question": question,
                "normalized": normalized,
                "retrieved": [
                    {
                        "chunkId": oid(r.chunk_id),
                        "documentId": oid(r.document_id),
                        "vectorRank": r.vector_rank,
                        "lexicalRank": r.lexical_rank,
                        "rrfScore": r.rrf_score,
                        "rerankScore": r.rerank_score,
                    }
                    for r in retrieved
                ],
                "usedChunkIds": [oid(c) for c in used_chunk_ids],
                "answer": answer,
                "refused": refused,
                "citations": [{"marker": c["marker"], "chunkId": oid(c["chunkId"])} for c in citations],
                "invalidCitations": invalid_citations,
                "timings": timings,
                "usage": usage,
                "feedback": None,
                "feedbackNote": "",
                "createdAt": now(),
            }
        )
        return str(res.inserted_id)

    async def find_by_id(self, workspace_id: str, query_id: str) -> dict[str, Any] | None:
        if not ObjectId.is_valid(query_id):
            return None
        return await self.col.find_one({"_id": oid(query_id), "workspaceId": oid(workspace_id)})

    async def set_feedback(self, workspace_id: str, query_id: str, feedback: str, note: str = "") -> None:
        await self.col.update_one(
            {"_id": oid(query_id), "workspaceId": oid(workspace_id)},
            {"$set": {"feedback": feedback, "feedbackNote": note}},
        )

    async def list_recent(self, workspace_id: str, limit: int = 50) -> list[dict[str, Any]]:
        return (
            await self.col.find({"workspaceId": oid(workspace_id)}).sort("createdAt", DESCENDING).limit(limit).to_list()
        )
