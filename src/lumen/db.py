from __future__ import annotations

from pymongo import ASCENDING, DESCENDING, TEXT, AsyncMongoClient
from pymongo.asynchronous.database import AsyncDatabase
from redis.asyncio import Redis

from lumen.config import Settings


def connect_mongo(settings: Settings) -> tuple[AsyncMongoClient, AsyncDatabase]:
    client: AsyncMongoClient = AsyncMongoClient(settings.mongo_uri, tz_aware=True)
    return client, client[settings.mongo_db_name]


def connect_redis(settings: Settings) -> Redis:
    return Redis.from_url(settings.redis_url, decode_responses=True)


async def ensure_indexes(db: AsyncDatabase) -> None:
    """Idempotent. Every chunk index leads with workspaceId because every retrieval path filters on it."""
    await db.workspaces.create_index("slug", unique=True)
    await db.workspaces.create_index("inviteCode", unique=True)
    await db.users.create_index("email", unique=True)
    await db.users.create_index("workspaceId")
    await db.documents.create_index("storageKey", unique=True)
    await db.documents.create_index([("workspaceId", ASCENDING), ("deletedAt", ASCENDING), ("createdAt", DESCENDING)])
    await db.documents.create_index([("workspaceId", ASCENDING), ("status", ASCENDING)])
    await db.chunks.create_index([("documentId", ASCENDING), ("chunkIndex", ASCENDING)])
    await db.chunks.create_index([("workspaceId", ASCENDING), ("live", ASCENDING)])
    # Lexical retrieval for VECTOR_BACKEND=local (Atlas uses a $search index instead).
    await db.chunks.create_index([("text", TEXT)], name="chunks_text")
    await db.queries.create_index([("workspaceId", ASCENDING), ("createdAt", DESCENDING)])
    await db.queries.create_index([("workspaceId", ASCENDING), ("feedback", ASCENDING)])
