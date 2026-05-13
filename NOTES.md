# NOTES

I used a small custom agent loop instead of LangChain's AgentExecutor. The trade-off is that I took on more responsibility for tool routing and provider handling, but gained very explicit control over streaming, retries, observability, and eval traces. With more tools or a production agent stack, I would revisit LangChain or LlamaIndex for standardized callbacks and ecosystem integrations.

The brief did not specify what "retrieval missed" should mean. I decided that non-empty vector results are not enough, so RAG answers must pass a relevance threshold; otherwise the tool returns `NO_RELEVANT_CONTEXT` and the agent refuses to guess.

The part I would extract into a shared agent toolkit is the observability wrapper: one place to record tool name, arguments, latency, success/failure, LLM token usage, and retrieval top-k scores as JSONL. That code would be useful for every future CLI agent, independent of this weather/RAG task.
