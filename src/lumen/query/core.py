"""Pure query-path logic, with no I/O: normalization, rank fusion, the tenant guard, prompt
assembly, citation validation and the refusal policy."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace

from lumen.ports import ChatMessage
from lumen.types import FusedChunk, RetrievedChunk, SourceRef
from lumen.utils import cosine_similarity, count_tokens, get_logger, truncate_to_tokens

log = get_logger("lumen.query")

ACRONYMS = {
    "MSA": "Master Services Agreement",
    "SOW": "Statement of Work",
    "NDA": "Non-Disclosure Agreement",
    "SLA": "Service Level Agreement",
    "DPA": "Data Processing Agreement",
}
_ACRONYM_PATTERN = re.compile(rf"\b({'|'.join(ACRONYMS)})\b", re.IGNORECASE)
MAX_QUERY_CHARS = 512


def normalize_query(question: str) -> str:
    """Embedding "MSA" alone lands nowhere near "Master Services Agreement" in vector space;
    the expanded form retrieves both."""
    collapsed = re.sub(r"\s+", " ", question.strip())
    expanded = _ACRONYM_PATTERN.sub(lambda m: f"{m.group(0)} ({ACRONYMS[m.group(0).upper()]})", collapsed)
    return expanded[:MAX_QUERY_CHARS]


def reciprocal_rank_fusion(lists: list[list[RetrievedChunk]], k: int = 60) -> list[FusedChunk]:
    """Reciprocal Rank Fusion. Cosine and text scores sit on incomparable scales, so RRF fuses on
    rank position instead of normalizing scores: a chunk ranked well by both retrievers outranks
    one ranked excellently by only one. List 0 is the vector list, list 1 the lexical list."""
    scores: dict[str, float] = {}
    ranks: dict[str, list[int | None]] = {}
    by_id: dict[str, RetrievedChunk] = {}
    for list_index, results in enumerate(lists):
        for i, chunk in enumerate(results):
            by_id.setdefault(chunk.chunk_id, chunk)
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + 1 / (k + i + 1)
            r = ranks.setdefault(chunk.chunk_id, [None, None])
            if list_index < 2 and r[list_index] is None:
                r[list_index] = i + 1
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [
        FusedChunk(
            **vars(by_id[cid]),
            vector_rank=ranks[cid][0],
            lexical_rank=ranks[cid][1],
            rrf_score=score,
        )
        for cid, score in ordered
    ]


class TenantLeakError(Exception):
    pass


def assert_no_tenant_leak(workspace_id: str, chunks: list[FusedChunk]) -> None:
    """Defense in depth. Cross-tenant results should be structurally impossible, because every
    retrieval path filters on workspaceId in the database. If a future index change or a dropped
    filter ever breaks that, this fails loudly instead of quietly returning another tenant's data."""
    leaked = [c.chunk_id for c in chunks if c.workspace_id != workspace_id]
    if leaked:
        log.critical("TENANT_LEAK workspace=%s leaked=%s", workspace_id, leaked)
        raise TenantLeakError(f"Tenant isolation check failed: {len(leaked)} chunk(s) from another workspace")


SYSTEM_PROMPT = (
    "Answer using ONLY the provided sources. Cite every claim as [n].\n"
    "If the sources do not contain the answer, say so explicitly.\n"
    "Never use knowledge outside the sources."
)


@dataclass
class PromptChunk:
    chunk: FusedChunk
    embedding: list[float]


def build_prompt(
    question: str,
    document_titles: dict[str, str],
    ranked: list[PromptChunk],
    token_budget: int,
    near_duplicate_threshold: float,
) -> tuple[list[ChatMessage], list[SourceRef]]:
    """Chunks are selected best-first, so the token budget and the near-duplicate skip favour the
    strongest ones, then rendered best-last: LLMs lose facts in the middle of long contexts, so the
    top-ranked chunk sits right before the question."""
    selected: list[PromptChunk] = []
    used = 0
    for pc in ranked:
        if any(cosine_similarity(s.embedding, pc.embedding) > near_duplicate_threshold for s in selected):
            continue
        tokens = count_tokens(pc.chunk.text)
        if used + tokens > token_budget:
            if not selected:  # always send at least one source, truncated to fit
                selected.append(
                    PromptChunk(replace(pc.chunk, text=truncate_to_tokens(pc.chunk.text, token_budget)), pc.embedding)
                )
            break
        selected.append(pc)
        used += tokens

    best_last = [pc.chunk for pc in reversed(selected)]
    sources = [
        SourceRef(
            marker=i + 1,
            chunk_id=c.chunk_id,
            document_id=c.document_id,
            document_title=document_titles.get(c.document_id, "Untitled document"),
            page_number=c.page_number,
            section=c.section,
            snippet=c.text[:240],
            bbox=c.bbox,
        )
        for i, c in enumerate(best_last)
    ]
    block = "\n\n".join(
        f"[{s.marker}] ({s.document_title}, p.{s.page_number}) {c.text}"
        for s, c in zip(sources, best_last, strict=True)
    )
    messages = [ChatMessage("system", SYSTEM_PROMPT), ChatMessage("user", f"Sources:\n{block}\n\nQuestion: {question}")]
    return messages, sources


@dataclass
class CitationCheck:
    cleaned_text: str
    citations: list[dict[str, object]]
    invalid_count: int


def validate_citations(raw_text: str, sources: list[SourceRef]) -> CitationCheck:
    """Every [n] the model emitted is checked against the real source list. A hallucinated [9]
    when only 8 sources were sent is stripped and counted, a direct grounding-failure signal."""
    valid = {s.marker: s.chunk_id for s in sources}
    seen: set[int] = set()
    invalid = 0

    def check(m: re.Match[str]) -> str:
        nonlocal invalid
        marker = int(m.group(1))
        if marker not in valid:
            invalid += 1
            return ""
        seen.add(marker)
        return m.group(0)

    cleaned = re.sub(r"\[(\d+)\]", check, raw_text)
    return CitationCheck(cleaned, [{"marker": m, "chunkId": valid[m]} for m in sorted(seen)], invalid)


REFUSAL_MESSAGE = "I couldn't find anything in your documents about this."


def should_refuse(top: FusedChunk | None, threshold: float) -> bool:
    """If the best reranked chunk scores below a calibrated threshold, skip generation: a fluent,
    plausible, wrong answer is worse than an admitted miss, and it saves the generation cost."""
    if top is None:
        return True
    score = top.rerank_score if top.rerank_score is not None else top.rrf_score
    return score < threshold
