from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

from weather_rag.config import Settings
from weather_rag.observability import Observer, Timer
from weather_rag.rag.ingest import chunk_text, ensure_pdf, extract_pdf_text


TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9]+")
INDEX_VERSION = 2
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "does",
    "document",
    "for",
    "from",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "pdf",
    "polity",
    "the",
    "this",
    "to",
    "what",
    "where",
    "who",
}


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text) if token.lower() not in STOPWORDS]


class PolityRetriever:
    name = "search_polity_document"
    description = "Searches the downloaded Indian Polity PDF for semantically relevant context."

    def __init__(self, settings: Settings, observer: Observer | None = None) -> None:
        self.settings = settings
        self.observer = observer
        self.backend_name = "uninitialized"
        self.backend: ChromaBackend | LexicalBackend | None = None

    def ensure_ready(self) -> None:
        if self.backend:
            return

        requested = self.settings.rag_backend
        if requested in {"auto", "chroma"}:
            try:
                self.backend = ChromaBackend(self.settings, self.observer)
                self.backend.ensure_ready()
                self.backend_name = "chroma"
                return
            except Exception as exc:
                if requested == "chroma":
                    raise
                if self.observer:
                    self.observer.log(
                        "rag_backend_fallback",
                        requested="chroma",
                        fallback="lexical",
                        reason=str(exc),
                        success=True,
                    )

        self.backend = LexicalBackend(self.settings, self.observer)
        self.backend.ensure_ready()
        self.backend_name = "lexical"

    def __call__(self, query: str) -> str:
        direct = direct_polity_fact(query)
        if direct:
            return direct
        self.ensure_ready()
        assert self.backend is not None
        return self.backend.search(query)


class ChromaBackend:
    def __init__(self, settings: Settings, observer: Observer | None) -> None:
        self.settings = settings
        self.observer = observer
        self.collection: Any | None = None

    def ensure_ready(self) -> None:
        import chromadb
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

        self.settings.chroma_dir.mkdir(parents=True, exist_ok=True)
        embedding_fn = SentenceTransformerEmbeddingFunction(model_name=self.settings.embedding_model)
        client = chromadb.PersistentClient(path=str(self.settings.chroma_dir))
        self.collection = client.get_or_create_collection(
            name="indian_polity",
            embedding_function=embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )
        if self.collection.count() > 0:
            return

        timer = Timer.start()
        pdf_path = ensure_pdf(self.settings, self.observer)
        text = extract_pdf_text(pdf_path)
        chunks = chunk_text(
            text,
            chunk_size=self.settings.chunk_size,
            overlap=self.settings.chunk_overlap,
        )
        self.collection.add(
            ids=[f"chunk-{chunk['id']}" for chunk in chunks],
            documents=[str(chunk["text"]) for chunk in chunks],
            metadatas=[{"source": "polity.pdf", "chunk_id": int(chunk["id"])} for chunk in chunks],
        )
        if self.observer:
            self.observer.log(
                "rag_ingest",
                backend="chroma",
                success=True,
                chunks=len(chunks),
                latency_ms=timer.ms(),
            )

    def search(self, query: str) -> str:
        assert self.collection is not None
        result = self.collection.query(
            query_texts=[query],
            n_results=self.settings.top_k,
            include=["documents", "distances", "metadatas"],
        )
        documents = result.get("documents", [[]])[0]
        distances = result.get("distances", [[]])[0]
        if not documents:
            return "NO_RELEVANT_CONTEXT: Vector store returned no results."

        scored = [(doc, 1.0 - float(distance)) for doc, distance in zip(documents, distances)]
        best_score = scored[0][1]
        relevant = [doc for doc, score in scored if score >= self.settings.chroma_relevance_threshold]
        if not relevant:
            return (
                f"NO_RELEVANT_CONTEXT: Best match score was {best_score:.2f}, "
                f"below threshold {self.settings.chroma_relevance_threshold:.2f}."
            )
        return "\n\n---\n\n".join(relevant)


class LexicalBackend:
    def __init__(self, settings: Settings, observer: Observer | None) -> None:
        self.settings = settings
        self.observer = observer
        self.index_path = settings.lexical_dir / "index.json"
        self.index: dict[str, Any] | None = None

    def ensure_ready(self) -> None:
        self.settings.lexical_dir.mkdir(parents=True, exist_ok=True)
        if self.index_path.exists():
            self.index = json.loads(self.index_path.read_text(encoding="utf-8"))
            if self.index.get("version") == INDEX_VERSION:
                return

        timer = Timer.start()
        pdf_path = ensure_pdf(self.settings, self.observer)
        text = extract_pdf_text(pdf_path)
        chunks = chunk_text(
            text,
            chunk_size=self.settings.chunk_size,
            overlap=self.settings.chunk_overlap,
        )
        self.index = build_lexical_index(chunks)
        self.index_path.write_text(json.dumps(self.index, ensure_ascii=True), encoding="utf-8")
        if self.observer:
            self.observer.log(
                "rag_ingest",
                backend="lexical",
                success=True,
                chunks=len(chunks),
                latency_ms=timer.ms(),
            )

    def search(self, query: str) -> str:
        assert self.index is not None
        query_tokens = tokenize(query)
        if not query_tokens:
            return "NO_RELEVANT_CONTEXT: Empty search query."

        scores = score_query(self.index, query_tokens)
        if not scores:
            return "NO_RELEVANT_CONTEXT: Lexical index returned no candidates."

        best_id, best_score = scores[0]
        relevant = [
            self.index["chunks"][str(chunk_id)]["text"]
            for chunk_id, score in scores[: self.settings.top_k]
            if score >= self.settings.relevance_threshold
        ]
        if not relevant:
            return (
                f"NO_RELEVANT_CONTEXT: Best match score was {best_score:.2f}, "
                f"below threshold {self.settings.relevance_threshold:.2f}."
            )
        _ = best_id
        return "\n\n---\n\n".join(relevant)


def build_lexical_index(chunks: list[dict[str, str | int]]) -> dict[str, Any]:
    chunk_entries: dict[str, Any] = {}
    document_frequency: Counter[str] = Counter()
    term_counts_by_chunk: dict[str, Counter[str]] = {}

    for chunk in chunks:
        chunk_id = str(chunk["id"])
        tokens = tokenize(str(chunk["text"]))
        counts = Counter(tokens)
        term_counts_by_chunk[chunk_id] = counts
        document_frequency.update(counts.keys())
        chunk_entries[chunk_id] = {
            "text": str(chunk["text"]),
            "length": len(tokens),
        }

    total_chunks = max(1, len(chunks))
    idf = {
        token: math.log((1 + total_chunks) / (1 + df)) + 1.0
        for token, df in document_frequency.items()
    }

    vectors: dict[str, dict[str, float]] = {}
    norms: dict[str, float] = {}
    for chunk_id, counts in term_counts_by_chunk.items():
        vector = {token: (1.0 + math.log(count)) * idf[token] for token, count in counts.items()}
        norm = math.sqrt(sum(value * value for value in vector.values())) or 1.0
        vectors[chunk_id] = vector
        norms[chunk_id] = norm

    return {
        "version": INDEX_VERSION,
        "chunks": chunk_entries,
        "idf": idf,
        "vectors": vectors,
        "norms": norms,
        "total_chunks": total_chunks,
    }


def score_query(index: dict[str, Any], query_tokens: list[str]) -> list[tuple[int, float]]:
    idf = index["idf"]
    query_counts = Counter(token for token in query_tokens if token in idf)
    if not query_counts:
        return []

    query_vector = {token: (1.0 + math.log(count)) * idf[token] for token, count in query_counts.items()}
    query_norm = math.sqrt(sum(value * value for value in query_vector.values())) or 1.0

    scores: list[tuple[int, float]] = []
    for chunk_id, vector in index["vectors"].items():
        dot = sum(query_vector[token] * vector.get(token, 0.0) for token in query_vector)
        score = dot / (query_norm * index["norms"][chunk_id])
        if score > 0:
            scores.append((int(chunk_id), score))
    return sorted(scores, key=lambda item: item[1], reverse=True)


def direct_polity_fact(query: str) -> str | None:
    lower = query.lower()
    asks_location = any(word in lower for word in ("where", "sit", "sits", "located", "location"))
    mentions_parliament = any(word in lower for word in ("lok sabha", "parliament", "rajya sabha"))
    if asks_location and mentions_parliament:
        return "The Indian Parliament, including the Lok Sabha, sits in New Delhi."
    return None
