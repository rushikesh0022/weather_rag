from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PlanAction:
    action: str
    tool: str | None = None
    arguments: dict[str, Any] = field(default_factory=dict)
    answer: str | None = None


@dataclass
class LLMCallResult:
    action: PlanAction | None
    content: str
    usage: dict[str, int | None]


@dataclass
class ToolObservation:
    tool: str
    arguments: dict[str, Any]
    result: str
    success: bool
    latency_ms: float
    error: str | None = None


@dataclass
class AgentResult:
    output: str
    trace: list[ToolObservation]
    usage: list[dict[str, Any]]
