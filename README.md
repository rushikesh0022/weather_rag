# Weather RAG CLI Agent

A command-line agent for the interview assignment. It can:

- answer Indian Polity questions from the provided Vision IAS PDF through RAG
- fetch current weather from Open-Meteo
- chain both tools in one turn, such as asking for the weather where the Lok Sabha sits
- stream only the final LLM answer in the CLI
- write JSONL observability records for LLM calls, tool calls, and weather API calls

The PDF asks for `gpt-4o`; this implementation also supports Gemini and DeepSeek because the local project notes target those providers. Set `LLM_PROVIDER=openai`, `gemini`, `deepseek`, or leave it as `auto`.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Add at least one key to `.env`:

```bash
DEEPSEEK_API_KEY=...
GEMINI_API_KEY=...
OPENAI_API_KEY=...
```

## Run The CLI

```bash
python main.py
```

Useful local/demo modes:

```bash
python main.py --mock-llm
python main.py --mock-llm --fake-weather
python main.py --rag-backend lexical
```

The first run downloads the Polity PDF and builds a local index under `data/`.

## Evaluation

```bash
python eval.py
```

By default, eval uses a deterministic mock LLM and fake weather so it is not blocked by paid API keys or Open-Meteo uptime. To exercise a real model:

```bash
python eval.py --real-llm --real-weather
```

Observability logs are written to:

```text
logs/observability.jsonl
logs/eval_observability.jsonl
```

Each record includes timestamp, event type, name/provider, latency, success/failure, and token usage for LLM calls when the provider returns it. Weather and RAG tool calls report latency and success but do not pretend to have token counts.
