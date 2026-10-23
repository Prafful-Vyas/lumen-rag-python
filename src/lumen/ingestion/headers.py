"""Contextual chunk headers: before embedding, every chunk is prefixed with one line that places
it in its parent document, e.g. "[Acme MSA 2024 - Section 4: Limitation of Liability]". Generated
in one batched LLM pass per 20 chunks, with the document outline as shared context."""

from __future__ import annotations

import json
import re

from lumen.ports import ChatMessage, CompletionProvider
from lumen.types import ChunkCandidate

BATCH_SIZE = 20


def deterministic_header(document_title: str, chunk: ChunkCandidate) -> str:
    return f"{document_title} - {chunk.section or f'page {chunk.page_number}'}"


class ContextualHeaderGenerator:
    def __init__(self, completion: CompletionProvider, use_llm: bool) -> None:
        self.completion = completion
        self.use_llm = use_llm

    async def generate(self, document_title: str, outline: str, chunks: list[ChunkCandidate]) -> list[str]:
        if not self.use_llm:
            return [deterministic_header(document_title, c) for c in chunks]
        headers: list[str] = []
        for start in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[start : start + BATCH_SIZE]
            generated = await self._generate_batch(document_title, outline, batch)
            headers += [h or deterministic_header(document_title, c) for h, c in zip(generated, batch, strict=True)]
        return headers

    async def _generate_batch(self, document_title: str, outline: str, batch: list[ChunkCandidate]) -> list[str]:
        listing = "\n\n".join(
            f'{i}. (section: "{c.section or "unknown"}", page {c.page_number})\n{c.text[:300]}'
            for i, c in enumerate(batch)
        )
        try:
            text, _ = await self.completion.complete(
                [
                    ChatMessage(
                        "system",
                        "For each numbered chunk below, write ONE short line (under 20 words) that situates it within "
                        'its parent document, in the style "Document Title - Section X: Section Name". Respond with '
                        "ONLY a JSON array of strings, same order and count as the input.",
                    ),
                    ChatMessage(
                        "user",
                        f"Document: {document_title}\n\nOutline:\n{outline or '(no outline available)'}\n\nChunks:\n{listing}",
                    ),
                ],
                max_tokens=1500,
                temperature=0.2,
            )
            match = re.search(r"\[[\s\S]*\]", text)
            parsed = json.loads(match.group(0) if match else text)
            if isinstance(parsed, list) and len(parsed) == len(batch):
                return [str(h) for h in parsed]
        except (ValueError, TypeError):
            pass
        return [deterministic_header(document_title, c) for c in batch]
