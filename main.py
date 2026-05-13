from __future__ import annotations

import argparse
import sys

from weather_rag.agent import Agent
from weather_rag.config import Settings
from weather_rag.llm import create_llm
from weather_rag.observability import Observer
from weather_rag.rag import PolityRetriever
from weather_rag.tools import WeatherTool


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CLI agent for Indian Polity RAG + current weather.")
    parser.add_argument("--mock-llm", action="store_true", help="Use the deterministic local planner/answerer.")
    parser.add_argument("--fake-weather", action="store_true", help="Use deterministic weather fixtures.")
    parser.add_argument("--no-stream", action="store_true", help="Do not stream the final answer in the CLI.")
    parser.add_argument("--quiet-tools", action="store_true", help="Hide tool and observability progress lines.")
    parser.add_argument("--log-file", default=None, help="Path for JSONL observability logs.")
    parser.add_argument("--rag-backend", choices=["auto", "chroma", "lexical"], default=None)
    parser.add_argument("--llm-provider", choices=["auto", "deepseek", "gemini", "openai", "mock"], default=None)
    return parser


def create_agent(args: argparse.Namespace) -> tuple[Agent, PolityRetriever, Settings]:
    settings = Settings.load(
        log_file=args.log_file,
        rag_backend=args.rag_backend,
        llm_provider=args.llm_provider,
    )
    settings.ensure_dirs()
    observer = Observer(settings.log_file, echo=not args.quiet_tools)
    llm = create_llm(settings, mock=args.mock_llm)
    weather_tool = WeatherTool(observer, fake=args.fake_weather)
    retriever = PolityRetriever(settings, observer)
    retriever.ensure_ready()
    agent = Agent(
        llm=llm,
        weather_tool=weather_tool,
        rag_tool=retriever,
        observer=observer,
        max_steps=settings.max_agent_steps,
    )
    return agent, retriever, settings


def main() -> int:
    args = build_parser().parse_args()
    try:
        agent, retriever, settings = create_agent(args)
    except Exception as exc:
        print(f"Startup error: {exc}", file=sys.stderr)
        return 1

    agent.observer.print_header(
        provider=agent.llm.provider,
        model=agent.llm.model,
        rag_backend=retriever.backend_name,
        log_file=settings.log_file,
    )

    while True:
        try:
            user_input = input("You > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            return 0
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye.")
            return 0
        try:
            agent.run(user_input, stream=not args.no_stream)
        except Exception as exc:
            print(f"Error: {exc}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
