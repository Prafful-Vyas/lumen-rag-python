"""Retrieval metrics. Read Recall@5 and Recall@20 as a diagnostic pair: high Recall@20 with low
Recall@5 means retrieval found the answer and the reranker buried it; both low means it was never
retrieved. Pass the pre-rerank fused list for Recall@20, the reranked list for the rest."""

import math


def recall_at_k(ranked_ids: list[str], gold: set[str], k: int) -> bool:
    return any(i in gold for i in ranked_ids[:k])


def reciprocal_rank(ranked_ids: list[str], gold: set[str]) -> float:
    return next((1 / (i + 1) for i, cid in enumerate(ranked_ids) if cid in gold), 0.0)


def ndcg_at_k(ranked_ids: list[str], gold: set[str], k: int) -> float:
    dcg = sum(1 / math.log2(i + 2) for i, cid in enumerate(ranked_ids[:k]) if cid in gold)
    idcg = sum(1 / math.log2(i + 2) for i in range(min(len(gold), k)))
    return dcg / idcg if idcg else 0.0


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
