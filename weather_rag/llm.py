from __future__ import annotations

import json
import os
import re
import time
from abc import ABC, abstractmethod
from typing import Any, Callable

from weather_rag.config import Settings
from weather_rag.prompts import FINAL_SYSTEM_PROMPT, PLANNER_SYSTEM_PROMPT
from weather_rag.schema import LLMCallResult, PlanAction


def render_transcript(transcript: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for item in transcript:
        role = item.get("role", "unknown")
        if role == "tool":
            lines.append(
                f"Tool {item.get('name')} called with {json.dumps(item.get('arguments', {}), ensure_ascii=True)} "
                f"returned:\n{item.get('content', '')}"
            )
        else:
            lines.append(f"{role.title()}: {item.get('content', '')}")
    return "\n\n".join(lines)


def extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
        if not match:
            raise ValueError(f"Planner did not return JSON: {text[:200]}")
        return json.loads(match.group(0))


def normalize_action(raw: dict[str, Any]) -> PlanAction:
    action = str(raw.get("action", "")).strip()
    if action == "call_tool":
        tool = str(raw.get("tool", "")).strip()
        args = raw.get("arguments") or {}
        if not isinstance(args, dict):
            raise ValueError("Planner arguments must be a JSON object")
        return PlanAction(action="call_tool", tool=tool, arguments=args)
    if action == "final":
        return PlanAction(action="final", answer=str(raw.get("answer", "")))
    raise ValueError(f"Unknown planner action: {action}")


def approximate_usage(prompt: str, completion: str) -> dict[str, int]:
    try:
        import tiktoken

        encoder = tiktoken.get_encoding("cl100k_base")
        prompt_tokens = len(encoder.encode(prompt))
        completion_tokens = len(encoder.encode(completion))
    except Exception:
        prompt_tokens = max(1, len(prompt.split()))
        completion_tokens = max(1, len(completion.split()))
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "estimated": True,
    }


class BaseLLM(ABC):
    provider: str
    model: str

    @abstractmethod
    def plan(self, transcript: list[dict[str, Any]]) -> LLMCallResult:
        raise NotImplementedError

    @abstractmethod
    def stream_final(self, transcript: list[dict[str, Any]], on_token: Callable[[str], None]) -> tuple[str, dict[str, Any]]:
        raise NotImplementedError


class OpenAICompatibleLLM(BaseLLM):
    def __init__(self, *, provider: str, model: str, api_key: str, base_url: str | None = None) -> None:
        from openai import OpenAI

        self.provider = provider
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def plan(self, transcript: list[dict[str, Any]]) -> LLMCallResult:
        prompt = render_transcript(transcript)
        messages = [
            {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
        }
        try:
            response = self.client.chat.completions.create(
                **kwargs,
                response_format={"type": "json_object"},
            )
        except Exception as exc:
            if not isinstance(exc, TypeError) and "response_format" not in str(exc).lower():
                raise
            response = self.client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content or ""
        usage = usage_from_openai(response)
        return LLMCallResult(action=normalize_action(extract_json_object(content)), content=content, usage=usage)

    def stream_final(self, transcript: list[dict[str, Any]], on_token: Callable[[str], None]) -> tuple[str, dict[str, Any]]:
        prompt = render_transcript(transcript)
        messages = [
            {"role": "system", "content": FINAL_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        usage: dict[str, Any] = {}
        chunks: list[str] = []
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
            "stream": True,
        }
        try:
            stream = self.client.chat.completions.create(
                **kwargs,
                stream_options={"include_usage": True},
            )
        except Exception as exc:
            if not isinstance(exc, TypeError) and "stream_options" not in str(exc).lower():
                raise
            stream = self.client.chat.completions.create(**kwargs)

        for event in stream:
            event_usage = getattr(event, "usage", None)
            if event_usage:
                usage = usage_from_openai_usage(event_usage)
            if not getattr(event, "choices", None):
                continue
            delta = event.choices[0].delta
            token = getattr(delta, "content", None)
            if token:
                chunks.append(token)
                on_token(token)
        answer = "".join(chunks)
        if not usage:
            usage = approximate_usage(FINAL_SYSTEM_PROMPT + "\n" + prompt, answer)
        return answer, usage


class GeminiLLM(BaseLLM):
    def __init__(self, *, model: str, api_key: str) -> None:
        import google.generativeai as genai

        self.provider = "gemini"
        self.model = model
        genai.configure(api_key=api_key)
        self._genai = genai

    def plan(self, transcript: list[dict[str, Any]]) -> LLMCallResult:
        prompt = PLANNER_SYSTEM_PROMPT + "\n\nTranscript:\n" + render_transcript(transcript)
        try:
            model = self._genai.GenerativeModel(
                self.model,
                generation_config={"temperature": 0, "response_mime_type": "application/json"},
            )
            response = model.generate_content(prompt)
        except Exception as exc:
            if "response_mime_type" not in str(exc).lower():
                raise
            model = self._genai.GenerativeModel(self.model, generation_config={"temperature": 0})
            response = model.generate_content(prompt)
        content = getattr(response, "text", "") or ""
        usage = usage_from_gemini(response, prompt, content)
        return LLMCallResult(action=normalize_action(extract_json_object(content)), content=content, usage=usage)

    def stream_final(self, transcript: list[dict[str, Any]], on_token: Callable[[str], None]) -> tuple[str, dict[str, Any]]:
        prompt = FINAL_SYSTEM_PROMPT + "\n\nTranscript:\n" + render_transcript(transcript)
        model = self._genai.GenerativeModel(self.model, generation_config={"temperature": 0})
        stream = model.generate_content(prompt, stream=True)
        chunks: list[str] = []
        usage: dict[str, Any] = {}
        for event in stream:
            token = getattr(event, "text", "") or ""
            if token:
                chunks.append(token)
                on_token(token)
            event_usage = usage_from_gemini(event, prompt, "".join(chunks), fallback=False)
            if event_usage:
                usage = event_usage
        answer = "".join(chunks)
        if not usage:
            usage = approximate_usage(prompt, answer)
        return answer, usage


class MockLLM(BaseLLM):
    provider = "mock"
    model = "deterministic"

    def plan(self, transcript: list[dict[str, Any]]) -> LLMCallResult:
        action = heuristic_plan(transcript)
        content = json.dumps(
            {
                "action": action.action,
                "tool": action.tool,
                "arguments": action.arguments,
                "answer": action.answer,
            },
            ensure_ascii=True,
        )
        return LLMCallResult(action=action, content=content, usage=approximate_usage(render_transcript(transcript), content))

    def stream_final(self, transcript: list[dict[str, Any]], on_token: Callable[[str], None]) -> tuple[str, dict[str, Any]]:
        answer = heuristic_final_answer(transcript)
        for part in re.findall(r"\S+\s*", answer):
            on_token(part)
            time.sleep(0.005)
        return answer, approximate_usage(render_transcript(transcript), answer)


def usage_from_openai(response: Any) -> dict[str, int | None]:
    return usage_from_openai_usage(getattr(response, "usage", None))


def usage_from_openai_usage(usage: Any) -> dict[str, int | None]:
    if not usage:
        return {}
    return {
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    }


def usage_from_gemini(response: Any, prompt: str, completion: str, *, fallback: bool = True) -> dict[str, Any]:
    metadata = getattr(response, "usage_metadata", None)
    if metadata:
        prompt_tokens = getattr(metadata, "prompt_token_count", None)
        completion_tokens = getattr(metadata, "candidates_token_count", None)
        total_tokens = getattr(metadata, "total_token_count", None)
        if total_tokens is not None:
            return {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            }
    return approximate_usage(prompt, completion) if fallback else {}


def create_llm(settings: Settings, *, mock: bool = False) -> BaseLLM:
    if mock or settings.llm_provider == "mock":
        return MockLLM()

    provider = settings.llm_provider
    deepseek_key = os.getenv("DEEPSEEK_API_KEY")
    gemini_key = os.getenv("GEMINI_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    if provider == "auto":
        if deepseek_key:
            provider = "deepseek"
        elif gemini_key:
            provider = "gemini"
        elif openai_key:
            provider = "openai"

    if provider == "deepseek" and deepseek_key:
        return OpenAICompatibleLLM(
            provider="deepseek",
            model=settings.deepseek_model,
            api_key=deepseek_key,
            base_url="https://api.deepseek.com",
        )
    if provider == "gemini" and gemini_key:
        return GeminiLLM(model=settings.gemini_model, api_key=gemini_key)
    if provider == "openai" and openai_key:
        return OpenAICompatibleLLM(
            provider="openai",
            model=settings.openai_model,
            api_key=openai_key,
        )

    raise RuntimeError(
        "No usable LLM provider found. Set DEEPSEEK_API_KEY, GEMINI_API_KEY, or OPENAI_API_KEY, "
        "or run with --mock-llm for a local deterministic demo."
    )


def heuristic_plan(transcript: list[dict[str, Any]]) -> PlanAction:
    user_text = next((item["content"] for item in transcript if item.get("role") == "user"), "")
    lower = user_text.lower()
    tool_events = [item for item in transcript if item.get("role") == "tool"]

    if not tool_events:
        if mentions_weather(lower) and mentions_polity_location(lower):
            return PlanAction(
                action="call_tool",
                tool="search_polity_document",
                arguments={"query": "Where does the Lok Sabha or Indian Parliament sit?"},
            )
        if mentions_weather(lower):
            return PlanAction(
                action="call_tool",
                tool="get_current_weather",
                arguments={"city": extract_city(user_text)},
            )
        if mentions_polity(lower) or "document" in lower or "pdf" in lower:
            return PlanAction(
                action="call_tool",
                tool="search_polity_document",
                arguments={"query": user_text},
            )
        return PlanAction(action="final", answer="outside scope")

    last_tool = tool_events[-1]
    if (
        mentions_weather(lower)
        and last_tool.get("name") == "search_polity_document"
        and "NO_RELEVANT_CONTEXT" not in last_tool.get("content", "")
    ):
        city = "New Delhi" if "new delhi" in last_tool.get("content", "").lower() else extract_city(last_tool.get("content", ""))
        return PlanAction(action="call_tool", tool="get_current_weather", arguments={"city": city})
    return PlanAction(action="final", answer="tool work complete")


def mentions_weather(text: str) -> bool:
    return any(word in text for word in ("weather", "temperature", "wind", "forecast"))


def mentions_polity_location(text: str) -> bool:
    return any(word in text for word in ("lok sabha", "rajya sabha", "parliament", "supreme court"))


def mentions_polity(text: str) -> bool:
    keywords = (
        "polity",
        "constitution",
        "fundamental right",
        "directive principle",
        "parliament",
        "lok sabha",
        "rajya sabha",
        "president",
        "supreme court",
        "citizen",
    )
    return any(word in text for word in keywords)


def extract_city(text: str) -> str:
    patterns = [
        r"\bweather\s+(?:in|at|for)\s+([A-Za-z .'-]+)",
        r"\btemperature\s+(?:in|at|for)\s+([A-Za-z .'-]+)",
        r"\bforecast\s+(?:in|at|for)\s+([A-Za-z .'-]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip(" ?.!,'\"")
    words = re.findall(r"[A-Za-z][A-Za-z .'-]+", text)
    return words[-1].strip(" ?.!,'\"") if words else text.strip()


def heuristic_final_answer(transcript: list[dict[str, Any]]) -> str:
    user_text = next((item["content"] for item in transcript if item.get("role") == "user"), "")
    tool_events = [item for item in transcript if item.get("role") == "tool"]
    if not tool_events:
        return "I can only answer questions about Indian Polity or current weather."

    last = tool_events[-1]
    content = last.get("content", "")
    if content.startswith("NO_RELEVANT_CONTEXT"):
        return "I could not find a reliable answer to that in the Polity document."
    if content.startswith("Error:"):
        return content
    if len(tool_events) >= 2 and mentions_weather(user_text.lower()):
        location = "New Delhi" if any("new delhi" in item.get("content", "").lower() for item in tool_events) else "the retrieved location"
        return f"The Polity document points to {location}. {content}"
    if last.get("name") == "get_current_weather":
        return content
    summary = content.split("\n\n---\n\n", 1)[0].strip()
    return "Based on the Polity document: " + summary[:900]
