from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


@dataclass
class Timer:
    started_at: float

    @classmethod
    def start(cls) -> "Timer":
        return cls(started_at=time.perf_counter())

    def ms(self) -> float:
        return round((time.perf_counter() - self.started_at) * 1000, 2)


class Observer:
    def __init__(self, path: Path, *, echo: bool = True) -> None:
        self.path = path
        self.echo = echo
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event: str, **fields: Any) -> dict[str, Any]:
        record = {"ts": utc_now(), "event": event, **fields}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")
        return record

    def print_tool_start(self, name: str, args: dict[str, Any]) -> None:
        if self.echo:
            print()
            print(f"> Tool: {name}")
            print(f"  args: {json.dumps(args, ensure_ascii=True)}")

    def print_tool_done(self, name: str, *, success: bool, latency_ms: float) -> None:
        if self.echo:
            state = "ok" if success else "failed"
            print(f"  status: {state} | latency: {latency_ms:.2f} ms")

    def print_retrieval_hits(self, hits: list[dict[str, Any]]) -> None:
        if not self.echo:
            return
        print("  retrieval top-k:")
        if not hits:
            print("    no candidates")
            return
        for hit in hits:
            rank = hit.get("rank", "?")
            score = hit.get("score")
            passed = "pass" if hit.get("passed") else "filtered"
            source = hit.get("source", "unknown")
            chunk_id = hit.get("chunk_id", "-")
            preview = str(hit.get("preview", "")).replace("\n", " ")
            score_text = f"{float(score):.4f}" if isinstance(score, int | float) else str(score)
            print(f"    {rank}. score={score_text} [{passed}] source={source} chunk={chunk_id}")
            if preview:
                print(f"       {preview}")

    def print_llm_usage(
        self,
        *,
        provider: str,
        model: str,
        phase: str,
        latency_ms: float,
        usage: dict[str, Any] | None,
    ) -> None:
        if not self.echo:
            return
        usage = usage or {}
        prompt = usage.get("prompt_tokens")
        completion = usage.get("completion_tokens")
        total = usage.get("total_tokens")
        if total is None:
            print()
            print(f"> LLM: {provider}:{model}")
            print(f"  phase: {phase} | latency: {latency_ms:.2f} ms | tokens: unreported")
            return
        print()
        print(f"> LLM: {provider}:{model}")
        print(
            f"  phase: {phase} | latency: {latency_ms:.2f} ms | "
            f"tokens: prompt={prompt}, completion={completion}, total={total}"
        )

    def print_header(self, *, provider: str, model: str, rag_backend: str, log_file: Path) -> None:
        if not self.echo:
            return
        print("Weather RAG CLI Agent")
        print("=====================")
        print(f"LLM          : {provider}:{model}")
        print(f"RAG backend  : {rag_backend}")
        print(f"Logs         : {log_file}")
        print("Commands     : exit, quit")
        print()
