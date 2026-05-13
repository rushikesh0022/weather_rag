from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - requirements install python-dotenv
    load_dotenv = None


ROOT_DIR = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    root_dir: Path = ROOT_DIR
    data_dir: Path = ROOT_DIR / "data"
    logs_dir: Path = ROOT_DIR / "logs"
    log_file: Path = ROOT_DIR / "logs" / "observability.jsonl"
    pdf_url: str = "https://cdn.visionias.in/value_added_material/5ca16-polity.pdf"
    pdf_path: Path = ROOT_DIR / "data" / "polity.pdf"
    chroma_dir: Path = ROOT_DIR / "data" / "chroma_store"
    lexical_dir: Path = ROOT_DIR / "data" / "lexical_store"
    rag_backend: str = "auto"
    chunk_size: int = 1200
    chunk_overlap: int = 200
    top_k: int = 5
    relevance_threshold: float = 0.16
    chroma_relevance_threshold: float = 0.35
    embedding_model: str = "all-MiniLM-L6-v2"
    llm_provider: str = "auto"
    deepseek_model: str = "deepseek-v4-flash"
    gemini_model: str = "gemini-2.5-flash"
    openai_model: str = "gpt-4o"
    max_agent_steps: int = 5

    @classmethod
    def load(
        cls,
        *,
        log_file: str | None = None,
        rag_backend: str | None = None,
        llm_provider: str | None = None,
    ) -> "Settings":
        if load_dotenv:
            load_dotenv(ROOT_DIR / ".env")

        logs_dir = ROOT_DIR / "logs"
        selected_log = Path(log_file).expanduser() if log_file else logs_dir / "observability.jsonl"

        return cls(
            log_file=selected_log if selected_log.is_absolute() else ROOT_DIR / selected_log,
            rag_backend=(rag_backend or os.getenv("RAG_BACKEND") or "auto").lower(),
            llm_provider=(llm_provider or os.getenv("LLM_PROVIDER") or "auto").lower(),
            deepseek_model=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
            gemini_model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
            openai_model=os.getenv("OPENAI_MODEL", "gpt-4o"),
        )

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
