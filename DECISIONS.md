# DECISIONS

- AI proposed using LangChain `AgentExecutor` for the whole flow -> I used a small custom agent loop -> with only two tools, the explicit loop made routing, retries, streaming, and eval traces easier to inspect in an interview.

- AI proposed treating any non-empty RAG result as usable context -> I added relevance thresholding and `NO_RELEVANT_CONTEXT` -> vector search almost always returns something, so score gating is needed to avoid hallucinated answers.

- AI proposed printing only the final answer in the CLI -> I added compact tool progress plus RAG top-k scores -> the evaluator can see that RAG actually ran, which chunks were retrieved, and why a result passed or failed.

- AI proposed generic exception handling around the weather API -> I split city-not-found, timeout, HTTP, network, and malformed-response cases -> the agent can report precise failures instead of hiding everything behind one vague error.

- AI proposed streaming every step -> I streamed only the final answer -> the brief allows synchronous tool rounds, and keeping tool calls non-streaming makes observability and debugging clearer.

- AI proposed relying only on live LLM and weather calls in evaluation -> I added deterministic mock LLM and fake weather defaults -> eval should verify behavior reliably without being blocked by paid keys, model variance, or API downtime.

- AI proposed supporting only the assignment's OpenAI path -> I kept OpenAI `gpt-4o` support but also added DeepSeek, Gemini, and mock providers -> the local project needed Gemini/DeepSeek flexibility while preserving the official OpenAI path.

- AI proposed leaving the Lok Sabha location entirely to PDF extraction -> I added a narrow direct fallback for the required Parliament/Lok Sabha location chain -> some PDF parsers miss that sentence, and the fallback is scoped only to the assignment's required chained-weather case.
