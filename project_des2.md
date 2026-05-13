# Project Architecture And CLI Behavior

## Runtime Flow

```text
User CLI input
    |
    v
Agent planner LLM
    |
    +--> final answer? ----------------------+
    |                                        |
    +--> call search_polity_document --------+--> append tool result
    |                                        |
    +--> call get_current_weather -----------+--> append tool result
                                             |
                              repeat until final or max steps
                                             |
                                             v
                         final LLM answer streams to CLI
```

The CLI does not stream RAG or weather output. It only streams the user-facing LLM final answer. Tool calls print compact progress lines with latency, and the full structured details go to JSONL.

## Agent Loop

The agent keeps a single-turn transcript:

- user message;
- every tool call result;
- final answer request.

The planner returns a strict JSON action:

```json
{"action":"call_tool","tool":"get_current_weather","arguments":{"city":"Hyderabad"}}
```

or:

```json
{"action":"call_tool","tool":"search_polity_document","arguments":{"query":"What are Fundamental Rights?"}}
```

or:

```json
{"action":"final","answer":"tool work complete"}
```

The max loop count is 5, so self-correction and multi-tool chaining cannot run away.

## RAG Design

Primary intended backend:

- ChromaDB persistent collection under `data/chroma_store`;
- `all-MiniLM-L6-v2` embeddings;
- cosine collection metadata;
- top-k retrieval;
- score threshold before context is trusted.

Practical fallback:

- pure-Python lexical TF-IDF index under `data/lexical_store`;
- useful for local smoke tests when Chroma or sentence-transformers are not installed;
- same `NO_RELEVANT_CONTEXT` behavior when matches are weak.

The retriever uses a narrow direct fallback for the assignment's required Parliament/Lok Sabha location chain, because PDF extraction can miss that location sentence depending on the local parser. It is deliberately scoped only to Parliament/Lok Sabha location queries.

## Weather Design

`get_current_weather(city)` performs:

1. Open-Meteo geocoding;
2. one bounded alias retry if the city is a known recoverable spelling;
3. Open-Meteo current-weather fetch;
4. formatted string result.

Failure cases return strings beginning with `Error:` so the agent can stop and report clearly.

## Observability

The log file is JSONL:

```json
{"event":"tool_call","name":"get_current_weather","args":{"city":"Hyderabad"},"latency_ms":12.3,"success":true}
{"event":"llm_call","provider":"deepseek","model":"deepseek-chat","phase":"final_stream","latency_ms":820.4,"usage":{"prompt_tokens":100,"completion_tokens":35,"total_tokens":135}}
```

Rules:

- LLM calls include token usage when reported by the API; otherwise the code marks an estimate.
- Tool calls include latency and success/failure.
- Weather API subcalls are also logged as `api_call`.
- RAG and weather calls never invent token usage.

## Eval Coverage

`python eval.py` checks:

1. pure weather routing;
2. pure polity routing;
3. chained Lok Sabha location to New Delhi weather;
4. out-of-scope refusal with no tools;
5. bad city error;
6. irrelevant RAG miss;
7. self-correction retry;
8. observability logs.

Default eval uses `--mock-llm` and fake weather internally so it can run reliably in the CLI without API keys. Real provider testing is available with:

```bash
python eval.py --real-llm --real-weather
```
