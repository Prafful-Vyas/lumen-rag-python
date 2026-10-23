"""Retrieval eval harness: ingests the labeled fixture documents fresh into a dedicated `lumen-eval`
workspace (through the real extract -> chunk -> embed -> index stages), then reports Recall@5,
Recall@20, MRR and nDCG@10. Gold chunks are resolved by (document title, text substring) because
chunk ids are generated at ingest time.

Usage: python -m eval.run_eval [--ablate-lexical | --ablate-vector]"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from bson import ObjectId

from eval.metrics import mean, ndcg_at_k, recall_at_k, reciprocal_rank
from lumen.composition import build_container
from lumen.config import get_settings
from lumen.query.services import RetrievalConfig, RetrievalService
from lumen.types import ChunkFilters

FIXTURE = Path(__file__).with_name("labeled-set.sample.json")
EVAL_WORKSPACE_SLUG = "lumen-eval"


class _Ablated:
    """Wraps a VectorIndex and blanks one retrieval path, to measure what each contributes."""

    def __init__(self, inner, drop: str) -> None:
        self.inner, self.drop = inner, drop

    async def vector_search(self, *args):
        return [] if self.drop == "vector" else await self.inner.vector_search(*args)

    async def lexical_search(self, *args):
        return [] if self.drop == "lexical" else await self.inner.lexical_search(*args)


async def main(ablate: str | None) -> list[dict[str, Any]]:
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    container = await build_container(get_settings(), queue="inline")
    try:
        workspace = await container.workspaces.find_by_slug(EVAL_WORKSPACE_SLUG) or await container.workspaces.create(
            "Lumen Eval"
        )
        workspace_id = str(workspace["_id"])
        user = await container.users.find_by_email("eval@lumen.local") or await container.users.create(
            "eval@lumen.local", "not-a-real-account", workspace_id, "admin"
        )

        # Wipe and re-ingest every run, so results always reflect the current pipeline code.
        await container.db.documents.delete_many({"workspaceId": workspace["_id"]})
        await container.db.chunks.delete_many({"workspaceId": workspace["_id"]})
        title_to_id: dict[str, str] = {}
        for d in fixture["documents"]:
            document_id = ObjectId()
            data = d["text"].encode("utf-8")
            key = f"eval/{document_id}.md"
            await container.storage.put(key, data, "text/markdown")
            await container.documents.create(
                document_id=document_id,
                workspace_id=workspace_id,
                title=d["title"],
                filename=f"{d['title']}.md",
                mime_type="text/markdown",
                storage_key=key,
                size_bytes=len(data),
                uploaded_by=str(user["_id"]),
            )
            await container.queue.enqueue("extract", str(document_id), workspace_id)
            title_to_id[d["title"]] = str(document_id)

        base = container.retrieval
        retrieval = RetrievalService(
            base.embedding,
            _Ablated(base.vector_index, ablate) if ablate else base.vector_index,
            base.reranker,
            base.query_cache,
            # rerank_top_k=10 so one reranked list covers both Recall@5 and nDCG@10.
            RetrievalConfig(
                base.config.vector_candidate_limit, base.config.lexical_candidate_limit, base.config.rrf_k, 10
            ),
        )

        results = []
        for q in fixture["questions"]:
            gold: set[str] = set()
            for g in q["gold"]:
                async for c in container.db.chunks.find({"documentId": ObjectId(title_to_id[g["documentTitle"]])}):
                    if g["textContains"] in c["text"]:
                        gold.add(str(c["_id"]))
            if not gold:
                print(f"  could not resolve a gold chunk, skipping: {q['question']}")
                continue
            r = await retrieval.retrieve(workspace_id, q["question"], ChunkFilters())
            fused_ids = [c.chunk_id for c in r.fused]
            reranked_ids = [c.chunk_id for c in r.reranked]
            results.append(
                {
                    "question": q["question"],
                    "recall5": recall_at_k(reranked_ids, gold, 5),
                    "recall20": recall_at_k(fused_ids, gold, 20),
                    "mrr": reciprocal_rank(reranked_ids, gold),
                    "ndcg10": ndcg_at_k(reranked_ids, gold, 10),
                    "retrieveMs": r.timings["retrieveMs"],
                }
            )
        report(results, container.settings.using_real_ai_providers, ablate)
        return results
    finally:
        await container.close()


def report(results: list[dict[str, Any]], real_ai: bool, ablate: str | None) -> None:
    mode = "openai" if real_ai else "mock embeddings (lexical path carries retrieval)"
    print(f"\nLumen retrieval eval: {len(results)} questions, providers={mode}, ablation={ablate or 'none'}\n")
    print(f"  Recall@5   : {mean([r['recall5'] for r in results]) * 100:.1f}%")
    print(f"  Recall@20  : {mean([r['recall20'] for r in results]) * 100:.1f}%")
    print(f"  MRR        : {mean([r['mrr'] for r in results]):.3f}")
    print(f"  nDCG@10    : {mean([r['ndcg10'] for r in results]):.3f}")
    print(f"  retrieve ms: {mean([r['retrieveMs'] for r in results]):.1f} (avg)\n")
    for r in results:
        flag = "OK  " if r["recall5"] else "MISS"
        print(f"  [{flag}] mrr={r['mrr']:.2f} ndcg10={r['ndcg10']:.2f}  {r['question']}")
    print("\nSynthetic 10-question sample set; extend eval/labeled-set.sample.json with real labeled data.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--ablate-lexical", action="store_const", const="lexical", dest="ablate")
    group.add_argument("--ablate-vector", action="store_const", const="vector", dest="ablate")
    asyncio.run(main(parser.parse_args().ablate))
