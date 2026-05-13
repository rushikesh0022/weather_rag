# Project Understanding From `AI_Interview_Assignment_v5.pdf`

## What The Assignment Is Asking Us To Build

This is a CLI-based agentic AI assistant with two grounded capabilities:

1. **Indian Polity RAG**
   - Programmatically download the provided Vision IAS Polity PDF.
   - Extract text, chunk it, index it locally, and retrieve top relevant chunks.
   - Expose the retriever as a callable tool named `search_polity_document`.
   - Avoid hallucination when retrieval returns irrelevant chunks. Non-empty search results are not enough; retrieval must pass a relevance gate.

2. **Current Weather Tool**
   - Accept a city name.
   - Call Open-Meteo geocoding to get latitude and longitude.
   - Call Open-Meteo forecast to get current weather.
   - Expose this as a callable tool named `get_current_weather`.
   - Handle city-not-found, timeout, HTTP, and malformed-response cases cleanly.

The important agent behavior is **tool routing plus multi-step chaining**. For a query like:

```text
What is the current weather where the Lok Sabha sits?
```

the agent must:

1. call `search_polity_document` first to find the location;
2. extract/use `New Delhi`;
3. call `get_current_weather(city="New Delhi")`;
4. return one final answer.

The agent must not be hard-coded for one tool call per turn. It needs a small loop with a max-step cap.

## Provider Decision

The PDF names OpenAI `gpt-4o`, but this local project is being built for Gemini or DeepSeek keys too. The code therefore supports:

- `DEEPSEEK_API_KEY` through an OpenAI-compatible client;
- `GEMINI_API_KEY` through Google Generative AI;
- `OPENAI_API_KEY` for assignment compatibility;
- `--mock-llm` for deterministic local/eval runs without paid keys.

Auto provider order is DeepSeek, Gemini, then OpenAI unless `LLM_PROVIDER` overrides it.

## Required Edge Cases

The PDF intentionally leaves edge-case behavior open. This project decides:

- Bad city: return a specific not-found message.
- Weather/geocoding timeout or HTTP failure: return the exact failure class.
- RAG miss: return `NO_RELEVANT_CONTEXT` when score gating fails.
- Out-of-scope question: use no tools and refuse with the limited scope.
- Chained weather query: RAG first, weather second.
- Self-correction stretch: retry one known city alias, such as `Bangaluru -> Bangalore`, then stop.

## Stretch Goals Implemented

1. **Streaming final responses**
   - Tool calls remain synchronous.
   - Only the final LLM answer streams token-by-token in the CLI.

2. **Per-call observability**
   - JSONL log records every LLM call, tool call, and weather API sub-call.
   - Logs include latency, success/failure, args, provider/model, and error when present.
   - Token usage is logged only for LLM calls, because RAG and weather APIs do not consume LLM tokens.

3. **Self-correction**
   - The weather tool performs one bounded alias retry for recoverable city spelling cases.

## Files Produced

```text
main.py                     CLI entrypoint
eval.py                     deterministic eval harness
requirements.txt            dependencies
.env.example                environment template
README.md                   run instructions
weather_rag/config.py       settings and paths
weather_rag/llm.py          DeepSeek/Gemini/OpenAI/mock provider layer
weather_rag/agent.py        multi-step planner/tool/final-answer loop
weather_rag/observability.py JSONL logging helpers
weather_rag/tools/weather.py Open-Meteo weather tool
weather_rag/rag/ingest.py   PDF download/extraction/chunking
weather_rag/rag/retriever.py Chroma or lexical retrieval with relevance gating
```

`NOTES.md` and `DECISIONS.md` are still part of the final PDF submission requirement, but they are intentionally not the focus yet because we are prioritizing a working CLI project first.
