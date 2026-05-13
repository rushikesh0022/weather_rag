from __future__ import annotations

from typing import Any, Callable

from weather_rag.llm import BaseLLM
from weather_rag.observability import Observer, Timer
from weather_rag.schema import AgentResult, ToolObservation


class Agent:
    def __init__(
        self,
        *,
        llm: BaseLLM,
        weather_tool: Callable[..., str],
        rag_tool: Callable[..., str],
        observer: Observer,
        max_steps: int = 5,
    ) -> None:
        self.llm = llm
        self.tools = {
            "get_current_weather": weather_tool,
            "search_polity_document": rag_tool,
        }
        self.observer = observer
        self.max_steps = max_steps

    def run(self, user_input: str, *, stream: bool = True) -> AgentResult:
        transcript: list[dict[str, Any]] = [{"role": "user", "content": user_input}]
        trace: list[ToolObservation] = []
        usage_records: list[dict[str, Any]] = []

        for _ in range(self.max_steps):
            timer = Timer.start()
            try:
                llm_result = self.llm.plan(transcript)
                latency_ms = timer.ms()
                usage_record = self._log_llm("planning", latency_ms, True, llm_result.usage, None)
                usage_records.append(usage_record)
            except Exception as exc:
                latency_ms = timer.ms()
                self._log_llm("planning", latency_ms, False, {}, str(exc))
                raise

            action = llm_result.action
            if not action or action.action == "final":
                break

            if action.action != "call_tool" or not action.tool:
                break
            observation = self._execute_tool(action.tool, action.arguments)
            trace.append(observation)
            transcript.append(
                {
                    "role": "tool",
                    "name": observation.tool,
                    "arguments": observation.arguments,
                    "content": observation.result,
                }
            )
        else:
            transcript.append(
                {
                    "role": "tool",
                    "name": "agent_loop",
                    "arguments": {},
                    "content": "Error: Agent stopped because max tool-call iterations were reached.",
                }
            )

        if stream:
            print("Assistant: ", end="", flush=True)

        timer = Timer.start()
        chunks: list[str] = []

        def on_token(token: str) -> None:
            chunks.append(token)
            if stream:
                print(token, end="", flush=True)

        try:
            answer, usage = self.llm.stream_final(transcript, on_token)
            latency_ms = timer.ms()
            if stream:
                print()
            usage_record = self._log_llm("final_stream", latency_ms, True, usage, None)
            usage_records.append(usage_record)
        except Exception as exc:
            latency_ms = timer.ms()
            self._log_llm("final_stream", latency_ms, False, {}, str(exc))
            raise

        return AgentResult(output=answer, trace=trace, usage=usage_records)

    def _execute_tool(self, tool_name: str, arguments: dict[str, Any]) -> ToolObservation:
        if tool_name not in self.tools:
            result = f"Error: Unknown tool '{tool_name}'."
            return ToolObservation(tool=tool_name, arguments=arguments, result=result, success=False, latency_ms=0.0, error=result)

        safe_args = normalize_tool_args(tool_name, arguments)
        self.observer.print_tool_start(tool_name, safe_args)
        timer = Timer.start()
        error = None
        try:
            result = self.tools[tool_name](**safe_args)
            success = not (result.startswith("Error:") or result.startswith("NO_RELEVANT_CONTEXT"))
        except Exception as exc:
            result = f"Error: {exc}"
            error = str(exc)
            success = False
        latency_ms = timer.ms()
        self.observer.log(
            "tool_call",
            name=tool_name,
            args=safe_args,
            latency_ms=latency_ms,
            success=success,
            error=error,
        )
        self.observer.print_tool_done(tool_name, success=success, latency_ms=latency_ms)
        return ToolObservation(
            tool=tool_name,
            arguments=safe_args,
            result=result,
            success=success,
            latency_ms=latency_ms,
            error=error,
        )

    def _log_llm(
        self,
        phase: str,
        latency_ms: float,
        success: bool,
        usage: dict[str, Any],
        error: str | None,
    ) -> dict[str, Any]:
        record = self.observer.log(
            "llm_call",
            provider=self.llm.provider,
            model=self.llm.model,
            phase=phase,
            latency_ms=latency_ms,
            success=success,
            usage=usage,
            error=error,
        )
        self.observer.print_llm_usage(
            provider=self.llm.provider,
            model=self.llm.model,
            phase=phase,
            latency_ms=latency_ms,
            usage=usage,
        )
        return record


def normalize_tool_args(tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    if tool_name == "get_current_weather":
        return {"city": str(arguments.get("city", "")).strip()}
    if tool_name == "search_polity_document":
        return {"query": str(arguments.get("query", "")).strip()}
    return arguments
