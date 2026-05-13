from __future__ import annotations

import re
from pathlib import Path

import requests

from weather_rag.config import Settings
from weather_rag.observability import Observer, Timer


def ensure_pdf(settings: Settings, observer: Observer | None = None) -> Path:
    settings.ensure_dirs()
    if settings.pdf_path.exists() and settings.pdf_path.stat().st_size > 0:
        return settings.pdf_path

    timer = Timer.start()
    response = requests.get(settings.pdf_url, timeout=60)
    response.raise_for_status()
    settings.pdf_path.write_bytes(response.content)
    if observer:
        observer.log(
            "api_call",
            name="polity_pdf_download",
            success=True,
            latency_ms=timer.ms(),
            url=settings.pdf_url,
            bytes=len(response.content),
        )
    return settings.pdf_path


def extract_pdf_text(pdf_path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        from PyPDF2 import PdfReader

    reader = PdfReader(str(pdf_path))
    parts: list[str] = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            parts.append(f"\n[page {page_number}]\n{text}")
    return "\n".join(parts)


def chunk_text(text: str, *, chunk_size: int, overlap: int) -> list[dict[str, str | int]]:
    cleaned = re.sub(r"[ \t]+", " ", text)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    chunks: list[dict[str, str | int]] = []
    start = 0
    chunk_id = 0
    while start < len(cleaned):
        end = min(len(cleaned), start + chunk_size)
        if end < len(cleaned):
            paragraph_break = cleaned.rfind("\n\n", start, end)
            sentence_break = cleaned.rfind(". ", start, end)
            boundary = max(paragraph_break, sentence_break)
            if boundary > start + chunk_size // 2:
                end = boundary + 1
        content = cleaned[start:end].strip()
        if content:
            chunks.append({"id": chunk_id, "text": content})
            chunk_id += 1
        if end >= len(cleaned):
            break
        start = max(0, end - overlap)
    return chunks
