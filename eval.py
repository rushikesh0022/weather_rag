from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from main import create_agent


@dataclass
class TestCase:
    name: str
    query: str
    check: Callable[[object, list[dict]], tuple[bool, str]]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate the Weather RAG CLI agent.")
    parser.add_argument("--real-llm", action="store_true", help="Use configured real LLM instead of mock.")
    parser.add_argument("--real-weather", action="store_true", help="Use real Open-Meteo instead of fake weather.")
    parser.add_argument("--rag-backend", choices=["auto", "chroma", "lexical"], default="lexical")
    parser.add_argument("--log-file", default="logs/eval_observability.jsonl")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    log_path = Path(args.log_file)
    if log_path.exists():
        log_path.unlink()

    agent_args = argparse.Namespace(
        mock_llm=not args.real_llm,
        fake_weather=not args.real_weather,
        no_stream=True,
        quiet_tools=True,
        log_file=args.log_file,
        rag_backend=args.rag_backend,
        llm_provider=None,
    )
    agent, _retriever, _settings = create_agent(agent_args)

    tests = make_tests()
    passed = 0
    print("Evaluation Report")
    print("=================")
    for test in tests:
        result = agent.run(test.query, stream=False)
        records = read_jsonl(log_path)
        ok, detail = test.check(result, records)
        passed += int(ok)
        tools = [obs.tool for obs in result.trace]
        print(f"{'PASS' if ok else 'FAIL'} | {test.name}")
        print(f"  query: {test.query}")
        print(f"  tools: {tools}")
        print(f"  answer: {one_line(result.output)}")
        print(f"  check: {detail}")

    total = len(tests)
    print(f"\nAggregate: {passed}/{total} passed")
    return 0 if passed == total else 1


def make_tests() -> list[TestCase]:
    return [
        TestCase(
            "pure weather",
            "What is the weather in Hyderabad?",
            lambda result, records: require_weather(result, "Hyderabad"),
        ),
        TestCase(
            "pure polity",
            "What are Fundamental Rights in India?",
            lambda result, records: require_tools(result, ["search_polity_document"], "rag called for polity"),
        ),
        TestCase(
            "chained weather through RAG",
            "What is the current weather where the Lok Sabha sits?",
            require_chained_new_delhi,
        ),
        TestCase(
            "out of scope",
            "What is 2 + 2?",
            require_no_tools,
        ),
        TestCase(
            "bad city edge case",
            "What is the weather in Zzzzxyz?",
            lambda result, records: require_output(result, "not found", "bad city reported"),
        ),
        TestCase(
            "irrelevant RAG miss",
            "In the polity document, who invented the telephone?",
            require_rag_miss,
        ),
        TestCase(
            "self-correction retry",
            "What is the weather in Bangaluru?",
            lambda result, records: require_output(result, "Retried with 'Bangalore'", "city alias retry surfaced"),
        ),
        TestCase(
            "observability records",
            "What is the weather in Hyderabad?",
            require_observability,
        ),
    ]


def require_tools(result: object, expected: list[str], detail: str) -> tuple[bool, str]:
    tools = [obs.tool for obs in result.trace]
    return tools == expected, f"{detail}; expected {expected}, got {tools}"


def require_weather(result: object, city: str) -> tuple[bool, str]:
    if not result.trace:
        return False, "weather tool was not called"
    obs = result.trace[0]
    ok = obs.tool == "get_current_weather" and obs.arguments.get("city", "").lower() == city.lower()
    return ok, f"expected get_current_weather city={city}, got {obs.tool} {obs.arguments}"


def require_chained_new_delhi(result: object, records: list[dict]) -> tuple[bool, str]:
    tools = [obs.tool for obs in result.trace]
    if tools[:2] != ["search_polity_document", "get_current_weather"]:
        return False, f"expected RAG then weather, got {tools}"
    city = result.trace[1].arguments.get("city", "")
    return city.lower() == "new delhi", f"weather city should be New Delhi, got {city}"


def require_no_tools(result: object, records: list[dict]) -> tuple[bool, str]:
    ok = not result.trace and "only answer" in result.output.lower()
    return ok, "expected no tool calls and an outside-scope refusal"


def require_output(result: object, needle: str, detail: str) -> tuple[bool, str]:
    return needle.lower() in result.output.lower(), detail


def require_rag_miss(result: object, records: list[dict]) -> tuple[bool, str]:
    if [obs.tool for obs in result.trace] != ["search_polity_document"]:
        return False, "expected exactly one RAG call"
    ok = "could not find" in result.output.lower() or "no_relevant_context" in result.trace[0].result.lower()
    return ok, "expected irrelevant retrieval to be rejected"


def require_observability(result: object, records: list[dict]) -> tuple[bool, str]:
    has_tool = any(record.get("event") == "tool_call" and "latency_ms" in record for record in records)
    has_llm = any(record.get("event") == "llm_call" and record.get("usage") for record in records)
    token_on_tool = any(record.get("event") == "tool_call" and "usage" in record for record in records)
    ok = has_tool and has_llm and not token_on_tool
    return ok, "expected tool latency logs, llm usage logs, and no fake tool token usage"


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def one_line(text: str) -> str:
    return " ".join(text.split())[:220]


if __name__ == "__main__":
    raise SystemExit(main())
