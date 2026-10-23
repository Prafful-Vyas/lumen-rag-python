"""Extraction strategies, picked by mime type. Every extractor returns per-page Markdown-ish
text (headings, lists and tables serialized where the format reveals them) plus a short outline
that later serves as shared context for contextual chunk headers."""

from __future__ import annotations

import io
import os
import re
from typing import Protocol

from lumen.types import ExtractedPage, ExtractionResult
from lumen.utils import get_logger

log = get_logger("lumen.extract")

PDF = "application/pdf"
DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
SUPPORTED_MIME_TYPES = (PDF, DOCX, "text/plain", "text/markdown")

SYNTHETIC_PAGE_CHARS = 3000
SPARSE_TEXT_CHARS_PER_PAGE = 20
HEADING_LINE = re.compile(r"^#{1,3}\s")


class Extractor(Protocol):
    def supports(self, mime_type: str) -> bool: ...

    def extract(self, data: bytes) -> ExtractionResult: ...


def synthetic_paginate(text: str) -> list[ExtractedPage]:
    """DOCX/TXT/MD have no fixed page geometry, so we paginate on paragraph boundaries at roughly
    a printed page's character budget. Citations then resolve to a stable, page-shaped unit. This
    is coarser than a real PDF page number, and is documented as such."""
    pages: list[ExtractedPage] = []
    buf = ""
    for para in re.split(r"\n{2,}", text):
        if buf and len(buf) + len(para) > SYNTHETIC_PAGE_CHARS:
            pages.append(ExtractedPage(len(pages) + 1, buf.strip()))
            buf = ""
        buf += ("\n\n" if buf else "") + para
    if buf.strip():
        pages.append(ExtractedPage(len(pages) + 1, buf.strip()))
    return pages or [ExtractedPage(1, text.strip())]


def _split_columns(line: str) -> list[str]:
    stripped = line.strip()
    return [c.strip() for c in re.split(r"\t+| {2,}", stripped) if c.strip()] if stripped else []


def serialize_tables_to_markdown(text: str) -> str:
    """Best-effort table detection over plain text: a run of 3+ consecutive lines that split into
    the same number (2+) of columns on tabs or 2+ spaces becomes a Markdown table, so row/column
    relationships survive into the embedding. Merged-cell tables still degrade."""
    lines = text.split("\n")
    out: list[str] = []
    i = 0
    while i < len(lines):
        first = _split_columns(lines[i])
        run = [first] if len(first) >= 2 else []
        j = i + 1
        while run and j < len(lines) and len(_split_columns(lines[j])) == len(first):
            run.append(_split_columns(lines[j]))
            j += 1
        if len(run) >= 3:
            header, *body = run
            out.append("| " + " | ".join(header) + " |")
            out.append("| " + " | ".join("---" for _ in header) + " |")
            out.extend("| " + " | ".join(r) + " |" for r in body)
            i = j
        else:
            out.append(lines[i])
            i += 1
    return "\n".join(out)


def _markdown_outline(text: str) -> str:
    return "\n".join([ln.strip() for ln in text.split("\n") if HEADING_LINE.match(ln.strip())][:60])


class TextExtractor:
    def supports(self, mime_type: str) -> bool:
        return mime_type in ("text/plain", "text/markdown")

    def extract(self, data: bytes) -> ExtractionResult:
        raw = data.decode("utf-8", errors="replace")
        is_markdown = raw.startswith("#") or "\n#" in raw
        text = raw if is_markdown else serialize_tables_to_markdown(raw)
        return ExtractionResult(synthetic_paginate(text), _markdown_outline(text))


class DocxExtractor:
    """DOCX tables are real structured XML, so they are converted to Markdown tables exactly rather
    than inferred heuristically. Body paragraphs and tables are walked in document order."""

    def supports(self, mime_type: str) -> bool:
        return mime_type == DOCX

    def extract(self, data: bytes) -> ExtractionResult:
        from docx import Document
        from docx.table import Table
        from docx.text.paragraph import Paragraph

        doc = Document(io.BytesIO(data))
        blocks: list[str] = []
        for child in doc.element.body.iterchildren():
            tag = child.tag.rsplit("}", 1)[-1]
            if tag == "p":
                blocks.append(self._paragraph(Paragraph(child, doc)))
            elif tag == "tbl":
                blocks.append(self._table(Table(child, doc)))
        markdown = "\n\n".join(b for b in blocks if b.strip())
        # Consecutive list items read better as one list block than as separate paragraphs.
        markdown = re.sub(r"(?m)^(- .+)\n\n(?=- )", r"\1\n", markdown)
        return ExtractionResult(synthetic_paginate(markdown), _markdown_outline(markdown))

    @staticmethod
    def _paragraph(p) -> str:
        text = p.text.strip()
        if not text:
            return ""
        style = (p.style.name if p.style is not None else "") or ""
        if m := re.match(r"Heading (\d)", style):
            return "#" * min(int(m.group(1)), 6) + " " + text
        if style == "Title":
            return "# " + text
        if "List" in style:
            return "- " + text
        return text

    @staticmethod
    def _table(table) -> str:
        rows = [[c.text.strip().replace("\n", " ").replace("|", "\\|") for c in r.cells] for r in table.rows]
        if not rows:
            return ""
        header, *body = rows
        lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
        lines += ["| " + " | ".join(r) + " |" for r in body]
        return "\n".join(lines)


class OcrFallback:
    """Renders one PDF page with pypdfium2 and OCRs it with Tesseract. Runs only on sparse pages,
    so a 200-page contract with 3 scanned appendix pages OCRs just those 3. Needs the tesseract
    binary on PATH (or TESSERACT_CMD); without it OCR is skipped and the extracted text kept."""

    def ocr_pdf_page(self, data: bytes, page_number: int) -> str | None:
        try:
            import pypdfium2 as pdfium
            import pytesseract

            if cmd := os.environ.get("TESSERACT_CMD"):
                pytesseract.pytesseract.tesseract_cmd = cmd
            pdf = pdfium.PdfDocument(data)
            try:
                image = pdf[page_number - 1].render(scale=2).to_pil()
            finally:
                pdf.close()
            return pytesseract.image_to_string(image)
        except Exception as exc:  # OCR is best-effort; never fail ingestion because of it.
            log.warning("OCR fallback failed for page %s (%s); keeping extracted text", page_number, exc)
            return None


class PdfExtractor:
    def __init__(self, ocr: OcrFallback | None = None) -> None:
        self.ocr = ocr or OcrFallback()

    def supports(self, mime_type: str) -> bool:
        return mime_type == PDF

    def extract(self, data: bytes) -> ExtractionResult:
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        pages: list[ExtractedPage] = []
        for i, page in enumerate(reader.pages, start=1):
            text, ocr = page.extract_text() or "", False
            if len(re.sub(r"\s+", "", text)) < SPARSE_TEXT_CHARS_PER_PAGE:
                ocr_text = self.ocr.ocr_pdf_page(data, i)
                if ocr_text and len(ocr_text.strip()) > len(text.strip()):
                    text, ocr = ocr_text, True
            pages.append(ExtractedPage(i, serialize_tables_to_markdown(text), ocr))
        return ExtractionResult(pages, self._outline(pages))

    @staticmethod
    def _outline(pages: list[ExtractedPage]) -> str:
        heading_like = re.compile(r"^(#{1,3}\s|[A-Z][A-Z0-9 .,'&-]{6,80}$|\d+(\.\d+)*\s+[A-Z])")
        lines = [
            f"p.{p.page_number}: {ln.strip()}"
            for p in pages
            for ln in [ln for ln in p.text.split("\n") if heading_like.match(ln.strip())][:5]
        ]
        return "\n".join(lines[:60])


class ExtractorRegistry:
    def __init__(self, extractors: list[Extractor] | None = None) -> None:
        self.extractors = extractors or [PdfExtractor(), DocxExtractor(), TextExtractor()]

    def extract(self, data: bytes, mime_type: str) -> ExtractionResult:
        for extractor in self.extractors:
            if extractor.supports(mime_type):
                return extractor.extract(data)
        if "presentation" in mime_type:
            raise ValueError("PPTX extraction is not supported yet (see README known gaps).")
        raise ValueError(f"Unsupported document type: {mime_type}")
