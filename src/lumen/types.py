"""Shared value types. Plain dataclasses so the domain layer has no framework dependency."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Literal

DocumentStatus = Literal["queued", "extracting", "chunking", "embedding", "ready", "failed"]
DOCUMENT_STATUSES: tuple[str, ...] = ("queued", "extracting", "chunking", "embedding", "ready", "failed")


@dataclass(frozen=True)
class BoundingBox:
    x: float
    y: float
    w: float
    h: float


@dataclass
class ChunkFilters:
    document_ids: list[str] | None = None
    folder_id: str | None = None
    after: datetime | None = None

    def cache_key(self) -> dict[str, Any]:
        return {
            "documentIds": sorted(self.document_ids) if self.document_ids else None,
            "folderId": self.folder_id,
            "after": self.after.isoformat() if self.after else None,
        }


@dataclass
class RetrievedChunk:
    """A chunk as it comes back from either retrieval path, before fusion. `score` is the raw
    retriever score (cosine or text score) and is not comparable across retrievers; see fusion.py."""

    chunk_id: str
    document_id: str
    workspace_id: str
    text: str
    page_number: int
    section: str
    score: float
    bbox: BoundingBox | None = None


@dataclass
class FusedChunk(RetrievedChunk):
    """A chunk after RRF fusion, carrying provenance from both retrievers for the query audit trail."""

    vector_rank: int | None = None
    lexical_rank: int | None = None
    rrf_score: float = 0.0
    rerank_score: float | None = None


@dataclass
class SourceRef:
    marker: int
    chunk_id: str
    document_id: str
    document_title: str
    page_number: int
    section: str
    snippet: str
    bbox: BoundingBox | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return {
            "marker": d["marker"],
            "chunkId": d["chunk_id"],
            "documentId": d["document_id"],
            "documentTitle": d["document_title"],
            "pageNumber": d["page_number"],
            "section": d["section"],
            "snippet": d["snippet"],
            "bbox": d["bbox"],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SourceRef:
        bbox = BoundingBox(**d["bbox"]) if d.get("bbox") else None
        return cls(
            marker=d["marker"],
            chunk_id=d["chunkId"],
            document_id=d["documentId"],
            document_title=d["documentTitle"],
            page_number=d["pageNumber"],
            section=d["section"],
            snippet=d["snippet"],
            bbox=bbox,
        )


@dataclass
class ExtractedPage:
    page_number: int
    # Markdown-ish plain text: headings, lists and tables serialized where the format reveals them.
    text: str
    ocr: bool = False


@dataclass
class ExtractionResult:
    pages: list[ExtractedPage]
    # Compressed structural outline, reused as shared context for contextual chunk headers.
    outline: str


@dataclass
class ChunkCandidate:
    page_number: int
    section: str
    text: str
    token_count: int


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    usd: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {"promptTokens": self.prompt_tokens, "completionTokens": self.completion_tokens, "usd": self.usd}


@dataclass
class SseEvent:
    """One server-sent event from the query endpoint: sources | token | done | error."""

    event: Literal["sources", "token", "done", "error"]
    data: dict[str, Any] = field(default_factory=dict)
