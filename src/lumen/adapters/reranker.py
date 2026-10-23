from __future__ import annotations

import json
import re

from lumen.ports import ChatMessage, CompletionProvider, RerankCandidate, RerankResult

CANDIDATE_TEXT_CHARS = 400


class LLMRerankerProvider:
    """Stand-in for a real cross-encoder: the completion model scores every (query, candidate)
    pair in one listwise call and returns JSON. Slower and costlier than a dedicated
    cross-encoder, and only as good as the model's instruction following. A real cross-encoder
    (local ONNX model or a rerank API) can replace it behind the same RerankerProvider port."""

    def __init__(self, completion: CompletionProvider) -> None:
        self.completion = completion

    async def rerank(self, query: str, candidates: list[RerankCandidate], top_k: int) -> list[RerankResult]:
        if not candidates:
            return []
        listing = "\n\n".join(f'{i}. id="{c.id}"\n{c.text[:CANDIDATE_TEXT_CHARS]}' for i, c in enumerate(candidates))
        text, _ = await self.completion.complete(
            [
                ChatMessage(
                    "system",
                    "You score how relevant each candidate passage is to the query, on a 0.0-1.0 scale. "
                    'Respond with ONLY a JSON array like [{"id":"...","score":0.0}, ...], one entry per candidate, no prose.',
                ),
                ChatMessage("user", f"Query: {query}\n\nCandidates:\n{listing}"),
            ],
            max_tokens=2000,
            temperature=0,
        )
        scored = parse_rerank_scores(text, candidates)
        return sorted(scored, key=lambda r: r.score, reverse=True)[:top_k]


def parse_rerank_scores(text: str, candidates: list[RerankCandidate]) -> list[RerankResult]:
    valid = {c.id for c in candidates}
    try:
        match = re.search(r"\[[\s\S]*\]", text)
        raw = json.loads(match.group(0) if match else text)
        by_id = {str(r["id"]): float(r["score"]) for r in raw if isinstance(r, dict) and str(r.get("id")) in valid}
        if by_id:
            return [RerankResult(c.id, by_id.get(c.id, 0.0)) for c in candidates]
    except (ValueError, TypeError, KeyError):
        pass
    # Unusable JSON: keep the fused order instead of failing the query.
    n = len(candidates)
    return [RerankResult(c.id, 1 - i / n) for i, c in enumerate(candidates)]


class MockReranker:
    """Keeps fused (RRF) order with decreasing scores. Zero cost, zero network."""

    async def rerank(self, query: str, candidates: list[RerankCandidate], top_k: int) -> list[RerankResult]:
        n = max(len(candidates), 1)
        return [RerankResult(c.id, 1 - i / n) for i, c in enumerate(candidates[:top_k])]
