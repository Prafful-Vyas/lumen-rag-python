"""Recursive structural chunker. Splits on the strongest available boundary first (heading,
then paragraph/table/list, then sentence, then a raw token window as a last resort) and never
splits a table or list mid-structure unless it is pathologically large."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from lumen.types import ChunkCandidate, ExtractedPage
from lumen.utils import count_tokens, decode, encode

TARGET_TOKENS = 512
OVERLAP_TOKENS = 64
HARD_MAX_TOKENS = 800
# Safety valve so one huge table or list can never exceed the embedding model's input limit.
ABSOLUTE_MAX_TOKENS = 2000

BlockType = Literal["heading", "table", "list", "paragraph"]
HEADING = re.compile(r"^#{1,6}\s+")
LIST_ITEM = re.compile(r"^(\s*[-*]|\s*\d+[.)])\s+")
LIST_CONTINUATION = re.compile(r"^(\s*[-*]|\s*\d+[.)])\s+|^\s{2,}\S")


@dataclass
class Block:
    type: BlockType
    text: str


def parse_blocks(page_text: str) -> list[Block]:
    lines = page_text.split("\n")
    blocks: list[Block] = []
    para: list[str] = []

    def flush_para() -> None:
        text = "\n".join(para).strip()
        if text:
            blocks.append(Block("paragraph", text))
        para.clear()

    i = 0
    while i < len(lines):
        line, stripped = lines[i], lines[i].strip()
        if not stripped:
            flush_para()
            i += 1
        elif HEADING.match(stripped):
            flush_para()
            blocks.append(Block("heading", stripped))
            i += 1
        elif stripped.startswith("|"):
            flush_para()
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append(lines[i].strip())
                i += 1
            blocks.append(Block("table", "\n".join(rows)))
        elif LIST_ITEM.match(line):
            flush_para()
            items = []
            while i < len(lines) and lines[i].strip() and LIST_CONTINUATION.match(lines[i]):
                items.append(lines[i].strip())
                i += 1
            blocks.append(Block("list", "\n".join(items)))
        else:
            para.append(line)
            i += 1
    flush_para()
    return blocks


def _token_window_split(text: str, block_type: BlockType, max_tokens: int) -> list[Block]:
    tokens = encode(text)
    return [Block(block_type, decode(tokens[i : i + max_tokens])) for i in range(0, len(tokens), max_tokens)]


def _split_paragraph_by_sentence(text: str) -> list[Block]:
    out: list[Block] = []
    buf = ""
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        if buf and count_tokens(f"{buf} {sentence}") > HARD_MAX_TOKENS:
            out.append(Block("paragraph", buf.strip()))
            buf = sentence
        else:
            buf = f"{buf} {sentence}" if buf else sentence
    if buf.strip():
        out.append(Block("paragraph", buf.strip()))
    return [
        piece
        for b in out
        for piece in (
            _token_window_split(b.text, "paragraph", HARD_MAX_TOKENS) if count_tokens(b.text) > HARD_MAX_TOKENS else [b]
        )
    ]


def expand_oversized_blocks(blocks: list[Block]) -> list[Block]:
    out: list[Block] = []
    for b in blocks:
        tokens = count_tokens(b.text)
        if b.type == "paragraph" and tokens > HARD_MAX_TOKENS:
            out.extend(_split_paragraph_by_sentence(b.text))
        elif b.type in ("table", "list") and tokens > ABSOLUTE_MAX_TOKENS:
            out.extend(_token_window_split(b.text, b.type, ABSOLUTE_MAX_TOKENS))
        else:
            out.append(b)
    return out


def _tail_tokens(text: str, n: int) -> str:
    tokens = encode(text)
    return "" if len(tokens) <= n else decode(tokens[-n:])


class StructuralChunker:
    def chunk_document(self, pages: list[ExtractedPage]) -> list[ChunkCandidate]:
        return [
            ChunkCandidate(page_number=page.page_number, section=section, text=text, token_count=tokens)
            for page in pages
            for section, text, tokens in self.chunk_page(page.text)
        ]

    def chunk_page(self, page_text: str) -> list[tuple[str, str, int]]:
        """Returns (section, text, token_count) per chunk."""
        chunks: list[tuple[str, str, int]] = []
        current_section = ""
        buffer: list[str] = []
        buffer_tokens = 0
        buffer_section = ""

        def flush() -> None:
            nonlocal buffer, buffer_tokens, buffer_section
            text = "\n\n".join(buffer).strip()
            if text:
                chunks.append((buffer_section or current_section, text, count_tokens(text)))
                overlap = _tail_tokens(text, OVERLAP_TOKENS)
                buffer = [overlap] if overlap else []
                buffer_tokens = count_tokens(overlap) if overlap else 0
                buffer_section = current_section
            else:
                buffer, buffer_tokens = [], 0

        for unit in expand_oversized_blocks(parse_blocks(page_text)):
            if unit.type == "heading":
                current_section = HEADING.sub("", unit.text).strip()
                buffer.append(unit.text)
                buffer_tokens += count_tokens(unit.text)
                buffer_section = buffer_section or current_section
                continue
            unit_tokens = count_tokens(unit.text)
            if buffer_tokens > 0 and buffer_tokens + unit_tokens > HARD_MAX_TOKENS:
                flush()
            buffer_section = buffer_section or current_section
            buffer.append(unit.text)
            buffer_tokens += unit_tokens
            if buffer_tokens >= TARGET_TOKENS:
                flush()
        flush()
        return chunks
