# Concepts & Interview Questions

A living reference for the concepts learned across this repo, paired with interview-style
questions and answers. Grounded in what's actually implemented here (specific files/functions
are cited) — not generic LangChain trivia. Organized to match the repo's own
[Suggested Learning Path](README.md#learning-path). Add to this over time as new scripts land;
keep it one evolving document, not dated sections (same convention as the README's
[Notes for Tomorrow](README.md#notes-for-tomorrow)).

---

## Table of Contents

1. [Chat Models, Messages & LCEL](#chat-models-messages--lcel)
2. [RAG Chain Shape (Fake Retriever)](#rag-chain-shape-fake-retriever)
3. [Real RAG (Embeddings + Vector Search)](#real-rag-embeddings--vector-search)
4. [Persistent RAG Ingestion (Crawl → Chunk → Embed → Upsert)](#persistent-rag-ingestion-crawl--chunk--embed--upsert)
5. [Tool Calling: `create_agent`, `bind_tools`, and Manual Loops](#tool-calling-create_agent-bind_tools-and-manual-loops)
6. [ReAct Agents / Hand-Rolled Agent Loops](#react-agents--hand-rolled-agent-loops)
7. [The Retrieval Tool + RAG Agent (`backend/core.py`)](#the-retrieval-tool--rag-agent-backendcorepy)
8. [Streamlit Integration (`app.py`)](#streamlit-integration-apppy)

---

## Chat Models, Messages & LCEL

**Concept.** `main.py` initializes `ChatOpenAI` and drives it two ways: raw message lists
(`SystemMessage` / `HumanMessage` / `AIMessage`) for multi-turn conversation, and a first LCEL
chain (`prompt2 | model | parser`) built from `ChatPromptTemplate.from_template`, `ChatOpenAI`,
and `StrOutputParser`. Multi-turn works by manually appending the prior `AIMessage` and a new
`HumanMessage` to the same list before the next `.invoke()` — the model has no memory of its
own; the caller *is* the memory. LCEL's `|` operator composes `Runnable`s so each step's output
becomes the next step's input, and the whole pipeline is itself a `Runnable` you can `.invoke()`.

### Interview Questions

1. **Q: In `main.py`, why does the second `llm.invoke(messages)` call "remember" the first
   exchange?**
   A: It doesn't, really — `ChatOpenAI` is stateless per call. The script manually appends the
   previous `AIMessage(content=response.content)` and the new `HumanMessage` onto the same
   `messages` list before invoking again, so the *full history* is resent as input every time.
   "Memory" here is just the caller keeping a growing list, not something the model retains.

2. **Q: What does the `|` operator actually do in `prompt2 | model | parser`?**
   A: Every LCEL component implements the `Runnable` interface. `|` calls `__or__`, which wraps
   both sides in a `RunnableSequence` — invoking the chain feeds the input into the first
   Runnable, passes its output as the input to the next, and so on. The composed chain is itself
   a `Runnable`, so it can be nested inside other chains (as seen later in `rag-tooling.py`'s
   `retrieval_step`).

3. **Q: Why put `StrOutputParser()` at the end of the chain instead of just reading
   `response.content`?**
   A: `model.invoke(...)` alone returns an `AIMessage` object. `StrOutputParser` unwraps that
   down to a plain string, so callers (and the next step in a longer chain) don't need to know
   about LangChain message types at all — it's the difference between "an object with more stuff
   attached" and "the actual text."

4. **Q: What's the difference between `SystemMessage`, `HumanMessage`, and `AIMessage`, and why
   does it matter which one you use when re-appending a model reply to history?**
   A: They're role-tagged message types the chat API uses to know who "said" what — system sets
   behavior/persona, human is the user, AI is the model's own prior output. Appending the model's
   reply back as `HumanMessage` instead of `AIMessage` would misrepresent the conversation to the
   model on the next turn (it would look like the user said it), which can distort how the model
   continues the conversation.

---

## RAG Chain Shape (Fake Retriever)

**Concept.** `rag-tooling.py` builds the *shape* of a RAG chain without spending any API calls on
retrieval: a `FakeRetriever.invoke()` hardcodes two `Document`s, `format_docs` joins them into one
string, and `RunnableParallel` runs a `context` branch (`RunnableLambda(retriever.invoke) |
RunnableLambda(format_docs)`) and a `question` branch (`RunnablePassthrough()`) on the *same*
input simultaneously, producing `{"context": ..., "question": ...}`. That dict feeds a prompt
whose `{context}`/`{question}` placeholders must match the `RunnableParallel` keys exactly.

### Interview Questions

1. **Q: What does `RunnableParallel` buy you here that running the branches sequentially
   wouldn't?**
   A: Both branches (`context` and `question`) receive the *same original input* independently —
   neither depends on the other's output — and the results are merged into one dict. Sequential
   composition (`|`) would force one branch's output to become the next's input, which is wrong
   when you need the raw question *and* a transformation of it side by side.

2. **Q: Why does `RunnableLambda` wrap `retriever.invoke` and `format_docs` instead of calling
   them directly?**
   A: Plain Python functions/methods aren't `Runnable`s — they don't support `|` or `.invoke()`
   as part of a chain. `RunnableLambda` adapts an arbitrary callable into a `Runnable` so it can
   be composed with `|` (as in `RunnableLambda(retriever.invoke) | RunnableLambda(format_docs)`)
   alongside real LangChain components.

3. **Q: If the prompt template used `{documents}` instead of `{context}`, what would happen at
   runtime, and why?**
   A: A `KeyError` when the prompt step runs — `ChatPromptTemplate.from_template` looks up each
   placeholder by name in the dict it receives, and `RunnableParallel` only produced `context`/
   `question` keys. The placeholder names and the `RunnableParallel` dict keys are coupled by
   naming convention only; nothing enforces they match except this at the point of failure.

4. **Q: This script never calls an embedding model or vector store API. Why is it still a useful
   step in the learning path (see `real_rag.py` next)?**
   A: It isolates the *chain wiring* (parallel branches → prompt → model → parser) from the
   *retrieval mechanics* (embeddings, similarity search). Once this shape is understood for free,
   `real_rag.py` only has to swap `FakeRetriever` for `vectorstore.as_retriever()` — same
   downstream chain, real search underneath.

---

## Real RAG (Embeddings + Vector Search)

**Concept.** `real_rag.py` replaces the fake retriever with a genuine pipeline:
`RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)` chunks a raw text blob into
`Document`s, `OpenAIEmbeddings(model="text-embedding-3-small")` embeds them into an
`InMemoryVectorStore`, and `vectorstore.as_retriever(search_kwargs={"k": 3})` performs real
cosine-similarity search at query time. The downstream chain shape
(`RunnableParallel → prompt → model → parser`) is identical to `rag-tooling.py` — only the
retriever swapped from hardcoded to real.

### Interview Questions

1. **Q: Why chunk the text at all before embedding it, instead of embedding the whole
   `RAW_TEXT` blob as one vector?**
   A: Embedding models have limited effective context, and one giant vector for a long,
   multi-topic document blurs together unrelated content, hurting retrieval precision. Smaller,
   focused chunks (`chunk_size=300` here) each capture one coherent idea, so a query's embedding
   can match the *specific* relevant chunk instead of a diluted whole-document vector.

2. **Q: What does `chunk_overlap=30` do, and why not set it to 0?**
   A: Overlap repeats the last N characters of one chunk at the start of the next, so a sentence
   or idea that falls right on a chunk boundary isn't split with no shared context between the two
   halves. Zero overlap risks cutting a relevant fact exactly at a boundary such that neither
   chunk alone retrieves well for a query about it.

3. **Q: How does `vectorstore.as_retriever()` decide which chunks are "relevant" to a question?**
   A: The question string is embedded with the *same* embedding model used to index the chunks,
   then the vector store compares that query vector against every stored chunk vector (cosine
   similarity by default) and returns the top-`k` closest ones — `k=3` here via
   `search_kwargs={"k": 3}`.

4. **Q: Why must the same embedding model be used for both indexing and querying?**
   A: Different embedding models place semantically similar text at different coordinates in
   different (and differently-dimensioned) vector spaces. A query vector from one model compared
   against document vectors from another is comparing incompatible spaces — the distances are
   meaningless. This exact constraint resurfaces in `ingestion.py`/`backend/core.py`, where both
   pin `OpenAIEmbeddings(model="text-embedding-3-large", dimensions=1024)` identically.

---

## Persistent RAG Ingestion (Crawl → Chunk → Embed → Upsert)

**Concept.** `ingestion.py` is the first script to build a RAG index meant to *survive* past one
run: a five-stage async pipeline — Crawl (`TavilyCrawl().ainvoke(...)`, BFS-style from a root URL,
`max_depth=2`) → filter (drop docs with no `raw_content`) → Chunk
(`RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)`, via `split_documents` so
`metadata={"source": url}` propagates to each chunk) → Embed
(`OpenAIEmbeddings(model="text-embedding-3-large", dimensions=1024)`, truncated from its native
3072 dims to match the fixed 1024-dim `langchain-doc-index` Pinecone index) → Upsert
(`index_documents_async`, batched and rate-limited with an `asyncio.Semaphore(MAX_CONCURRENT_UPSERTS=3)`
around `vectorstore.aadd_documents(batch)`, gathered via `asyncio.gather`).

### Interview Questions

1. **Q: Why does the batch-building loop in `index_documents_async` currently produce a "sliding
   window" instead of disjoint batches, and what's the practical effect?**
   A: It's written as `for i in range(0, len(documents)): batches.append(documents[i:i+UPSERT_BATCH_SIZE])`
   — missing the step argument, so `range` defaults to step 1. Each iteration advances by one
   document instead of `UPSERT_BATCH_SIZE` (100), so consecutive batches overlap almost entirely.
   The fix is `range(0, len(documents), UPSERT_BATCH_SIZE)`. Practically, every chunk gets
   upserted up to ~100 times, multiplying Pinecone writes and cost for no benefit — this is the
   repo's currently tracked open bug.

2. **Q: What does `asyncio.Semaphore(MAX_CONCURRENT_UPSERTS)` actually limit, given that all
   batch coroutines are created up front and handed to `asyncio.gather` at once?**
   A: It doesn't limit how many *tasks exist* — all of them are created immediately. It limits how
   many are allowed *inside* the `async with semaphore:` block (i.e., actually calling Pinecone)
   at the same time; the rest simply await acquiring the semaphore before proceeding. Task
   creation and concurrency throttling are separate concerns here.

3. **Q: Why does each `upsert_batch` call wrap `vectorstore.aadd_documents(batch)` in its own
   `try/except` instead of letting exceptions propagate up to the `asyncio.gather(*tasks)` call?**
   A: `asyncio.gather`'s default behavior is to re-raise the first exception it sees and cancel
   every other in-flight task. Catching and logging inside each task means no exception ever
   reaches `gather`, so one bad batch is skipped without taking down every other batch's upsert.

4. **Q: `TavilyCrawl` has `handle_tool_error=True` by default. What does that mean for how
   `res["results"]` is used, and what bug would skipping the `isinstance(res, dict)` check
   produce?**
   A: With `handle_tool_error=True`, an internal tool failure is returned as a plain **string**
   error message instead of raising an exception. If the code indexes into it as though it's still
   the expected dict shape (`res["results"]`), Python raises `TypeError: string indices must be
   integers` — a confusing error that hides the real underlying failure message. Guarding with
   `isinstance(res, dict)` (and printing `res` otherwise) surfaces the actual problem.

5. **Q: Why does `Document(page_content=doc.get("raw_content"))` need to be guarded with
   `if doc.get("raw_content")` before construction?**
   A: Not every crawled URL yields extractable content — redirects, non-HTML assets, or failed
   extraction can return `raw_content: None`. `Document.page_content` is a required string field
   in its pydantic schema, so constructing one with `None` raises a `ValidationError`. Filtering
   first avoids building `Document`s for URLs that had nothing usable to extract.

6. **Q: `RecursiveCharacterTextSplitter(chunk_size=4000)` and `OpenAIEmbeddings(chunk_size=50)`
   both use the parameter name `chunk_size` in the same file. Are they the same concept?**
   A: No — completely different axes. The text splitter's `chunk_size=4000` is a character-length
   cap per text chunk (controls retrieval granularity). The embeddings client's `chunk_size=50` is
   how many texts get batched into a single OpenAI embeddings API call (a throughput/rate-limit
   tradeoff, unrelated to text length). Same word, two unrelated concerns a few lines apart.

---

## Tool Calling: `create_agent`, `bind_tools`, and Manual Loops

**Concept.** This track has five scripts that progressively strip away abstraction around the
same job-search tools (`get_jobs`/`get_job_details` or `get_new_jobs`) to show what each layer
does:

- `tool_calling.py` — `create_agent(model, tools=[...], system_prompt=...)`, free-text output.
- `tool_calling_with_pydantic_schema.py` — same idea, but `response_format=AgentResponse`
  (a Pydantic model) makes `create_agent` return `result["structured_response"]` as a typed
  object instead of raw text.
- `tool_calling_manual.py` — `create_agent` removed; `model.bind_tools(TOOLS)` plus a
  hand-written loop: invoke → if `response.tool_calls`, run each tool and append a `ToolMessage`
  keyed by `tool_call_id` → invoke again → repeat until no tool calls remain, capped by
  `MAX_STEPS`.
- `teach_tool_calling.py` — the same manual loop stripped to one tool, isolating the `ToolCall`
  dict shape (`{name, args, id, type}`) and how `BaseTool.invoke()` branches on whether it's
  handed just `args` (returns the raw value) or the whole `ToolCall` (returns a `ToolMessage`
  tagged with `tool_call_id`).
- `tool_calling_manual_pydantic.py` — the tool is a bare Pydantic `BaseModel` (`GetNewJobs`), no
  `@tool` — `bind_tools` accepts the class directly and produces the identical `tool_calls`
  shape, but with no `BaseTool` there's no `.invoke()`; execution (`FUNCTIONS_BY_NAME[name](**args)`)
  and `ToolMessage` construction are both written by hand.

### Interview Questions

1. **Q: What is `create_agent`'s tool loop actually doing under the hood, in terms of the
   primitives shown in `tool_calling_manual.py`?**
   A: Bind tools to the model → invoke → if the response has `tool_calls`, run each one and
   append a `ToolMessage` matched back by `tool_call_id` → invoke again → repeat until a response
   comes back with no tool calls. `create_agent` runs this as a LangGraph state graph internally,
   which is also where it gets streaming, checkpointing, and `response_format` coercion —
   none of which the manual version has.

2. **Q: In `teach_tool_calling.py`, what's the difference between calling
   `tool_fn.invoke(tool_call["args"])` and `tool_fn.invoke(tool_call)` (the whole dict)?**
   A: `BaseTool.invoke()` inspects the *shape* of its input to decide behavior — there's no
   explicit mode flag. Passing just the args dict runs the tool and returns its raw return value.
   Passing the full `ToolCall` dict (`{name, args, id, type}`) makes it detect that shape, run the
   function using `args`, and wrap the output in a `ToolMessage` carrying `tool_call_id` — which
   is required for the result to be matched back to the correct call once resent to the model.

3. **Q: `tool_calling_manual_pydantic.py` binds a bare `BaseModel` (no `@tool`) via
   `model.bind_tools([GetNewJobs])`. Why does this still work, and what do you lose compared to
   `@tool`?**
   A: `bind_tools` only cares about producing a valid tool-call schema for the model to request
   against — it doesn't care whether that schema came from a `@tool`-decorated function or a plain
   Pydantic class; the resulting `tool_calls` have the identical `{name, args, id, type}` shape
   either way. What you lose is a `BaseTool` object: no `.invoke()`, no automatic `ToolMessage`
   wrapping — both have to be written by hand (`FUNCTIONS_BY_NAME[tool_call["name"]](**tool_call["args"])`,
   then a manual `ToolMessage(...)`).

4. **Q: Why does `get_new_jobs(**tool_call["args"])` work but `tool_fn.invoke(**tool_call["args"])`
   would raise a `TypeError`?**
   A: A plain Python function needs its arguments spread into named parameters — `**tool_args`
   does exactly that. `BaseTool.invoke()` has a single positional `input` parameter (str, dict, or
   `ToolCall`) that it pattern-matches on internally; unpacking a dict into keyword arguments for
   it fails because `invoke()` isn't expecting keyword args matching the tool's schema, it's
   expecting one object. Same underlying data, opposite calling convention.

5. **Q: Why would forgetting `messages.append(response)` (the `AIMessage` with `tool_calls`)
   before appending the resulting `ToolMessage`s break the next `model_with_tools.invoke(messages)`
   call?**
   A: OpenAI's API enforces that a `role: "tool"` message must directly follow an assistant
   message containing the matching `tool_calls` id — it's effectively a bracket-matching rule
   (the `AIMessage` opens each call id, the `ToolMessage` closes it). Skip the `AIMessage`, and the
   API 400s with "messages with role 'tool' must be a response to a preceding message with
   'tool_calls'" because there's no open bracket for the `ToolMessage` to close.

6. **Q: What real capability does a free-form `query: str` parameter (as in `get_new_jobs`) give
   the agent that a rigid `location: str`-only tool signature wouldn't?**
   A: It lets the model compose its actual intent into one string instead of being constrained to
   whatever fixed fields the tool exposes. The repo's own learning notes observed this concretely:
   a `location`-only tool couldn't express "roles at Meta, Google, Salesforce, Uber," so the agent
   called it four times with the same input, unable to encode what it wanted — a free-form query
   fixed that in one call.

---

## ReAct Agents / Hand-Rolled Agent Loops

**Concept.** `teach_react_agent.py` and `agent_loop_with_react_prompt.py` implement ReAct
(Reason+Act) without `bind_tools` or `create_agent` at all — the model never sees a structured
tool schema. Instead, a text-format prompt (`Thought` / `Action` / `Action Input` / `Observation`
/ `Final Answer`) is spelled out, and `model.invoke(prompt, stop=["\nObservation:"])` cuts
generation off right before the model could hallucinate its own fake result. The loop formats the
prompt with a running `scratchpad`, checks for `"Final Answer:"` in the reply, otherwise parses
out the action (and, with two tools, the action *name*) and input via string ops, runs the real
tool, and appends `reply + Observation` back onto the scratchpad for the next iteration.
`teach_react_agent.py` has one tool (nothing to choose); `agent_loop_with_react_prompt.py` adds a
second tool (`JobSearchTool`, `CompanyInfoTool`), forcing the prompt to list both via
`TOOLS_DESCRIPTIONS` and the parsing to recover the chosen action name, not just its input.

### Interview Questions

1. **Q: Why does ReAct not need `model.bind_tools(...)` at all, unlike the tool-calling track?**
   A: `bind_tools` exists to get the model to emit a structured `tool_calls` field the API
   understands. ReAct instead teaches the model a *text convention* — the loop parses that text
   with plain string operations (`reply.split("Action Input:")[-1].strip()`, etc.) to recover the
   same "which action, what input" information a `ToolCall` would carry structurally. Same
   request → execute → respond loop, driven by string parsing instead of a schema.

2. **Q: What does `stop=["\nObservation:"]` protect against, and what would happen without it?**
   A: It cuts the model's generation off exactly where the real tool result should be inserted.
   Without it, the model — trained to complete plausible-looking text — will happily write its own
   fabricated `Observation:` line instead of waiting for the actual tool call to run, breaking the
   loop's grounding in real data.

3. **Q: In `agent_loop_with_react_prompt.py`, why can `Action Input` still be extracted with
   `reply.split("Action Input:")[-1].strip()`, but `Action` needs
   `reply.splitlines()` + `line.startswith("Action:")` first?**
   A: `Action Input` is the last field before the `stop` sequence cuts generation, so nothing
   trails it — splitting on the label and taking everything after it is safe. `Action` is
   followed by `Action Input:` on the next line, so grabbing "everything after `Action:`" directly
   would swallow the `Action Input` line too. Isolating to the single line containing `Action:`
   first, then splitting on that line only, avoids the overlap.

4. **Q: `agent_loop_with_react_prompt.py`'s tools are defined as `BaseModel` subclasses
   (`JobSearchTool`, `CompanyInfoTool`) but never passed through `bind_tools`. What are they
   actually used for, and why read `.model_fields["description"].default` instead of
   `JobSearchTool.description`?**
   A: They're used purely to hold a name + description that gets read back into the prompt text
   (`TOOLS_DESCRIPTIONS`), so the model has real tool descriptions to choose between — there's no
   schema binding or validation happening. Pydantic v2 model fields don't exist as class-level
   attributes (`JobSearchTool.description` raises `AttributeError`, since fields are per-instance
   data); `model_fields` is a dict of `FieldInfo` objects that *is* available on the class itself,
   so `.model_fields["description"].default` is how you read a field's default without
   instantiating the model.

5. **Q: Hand-rolled ReAct loops have no built-in repetition guard. What actually happened in this
   repo when that surfaced, and how was it fixed?**
   A: Given a two-step task ("find a job at Salesforce, then describe its culture"), the model
   sometimes retried an identical `Action`/`Action Input` pair for several steps, apologizing each
   time, and ran out of its step budget before reaching `Final Answer` — nothing like
   `create_agent`'s LangGraph loop stops that on its own. It was fixed with two independent
   changes: an explicit prompt line ("do not repeat an Action with the same Action Input you've
   already tried") and raising the step budget from 6 to 10. A tracked follow-up (see README
   Notes for Tomorrow) is to add a *code-level* guard — tracking seen `(action, input)` pairs —
   rather than relying on the model to follow the instruction.

6. **Q: The repo confirmed the model "genuinely chooses" in `agent_loop_with_react_prompt.py`.
   What was the actual evidence for that, rather than just assuming the two-tool prompt works?**
   A: Given "find a job at Salesforce, then tell me about Salesforce's culture," the model called
   `JobSearchTool` first, judged the result insufficient for the second half of the question, and
   switched to `CompanyInfoTool` on its own — driven only by reading `TOOLS_DESCRIPTIONS` in the
   prompt, with no code forcing that sequence. Observed behavior, not just a prompt that looks
   like it should work.

---

## The Retrieval Tool + RAG Agent (`backend/core.py`)

**Concept.** `backend/core.py` is the read side of the pipeline `ingestion.py` writes into — the
first script in the repo that actually queries `langchain-doc-index`. `retrieve_context` is
decorated `@tool(response_format="content_and_artifact")`, which changes its return contract from
one value to a `(content, artifact)` tuple: `content` is the serialized "Source: ... Content: ..."
string the model reads, and `artifact` is the raw list of retrieved `Document`s. `run_llm(query)`
wires `retrieve_context` into `create_agent(model, tools=[retrieve_context], system_prompt=...)`
and calls `.invoke({"messages": [...]})`, which returns a **state dict**
(`{"messages": [...]}`), not a `{"content": ...}` shape — so the final answer is
`response["messages"][-1].content`. Because `response_format="content_and_artifact"` attaches the
raw `Document`s as `.artifact` on the `ToolMessage` the agent produces mid-run, `run_llm` walks
`response["messages"]`, finds `ToolMessage` instances named `"retrieve_context"`, and reshapes each
`Document`'s `.artifact` into `{"source": ..., "content": ...}` — returning
`{"answer": ..., "context": [...]}` instead of just the text answer, without a second retrieval
call.

### Interview Questions

1. **Q: What does `@tool(response_format="content_and_artifact")` change about what
   `retrieve_context` is allowed to return, and why use it here instead of a plain `@tool`?**
   A: A normal `@tool` returns one value that becomes `ToolMessage.content` directly. With
   `response_format="content_and_artifact"`, the function must return a `(content, artifact)`
   tuple — `content` (the serialized string) is what the model reads and reasons over, while
   `artifact` (the raw `Document` list) is preserved for other code to use directly. It's used
   here specifically so the retrieved `Document`s — with their `source` metadata — survive past
   the point the model consumes them as text, enabling citations without re-querying the vector
   store.

2. **Q: Why does `run_llm` return `response["messages"][-1].content` instead of something like
   `response["content"]` or `response["answer"]`?**
   A: `create_agent(...).invoke(...)` builds and runs a LangGraph state graph under the hood, so
   `.invoke()` returns whatever's in that graph's state — a dict, and specifically `messages` is
   the key holding the running conversation. There's no `"content"` or `"answer"` key; the final
   answer is just the last message in that list.

3. **Q: Concretely, how does `run_llm` recover the `Document`s behind an answer without
   re-running retrieval?**
   A: It iterates `response["messages"]`, checking for `isinstance(msg, ToolMessage) and
   msg.name == "retrieve_context" and msg.artifact` — that `.artifact` attribute is exactly the
   raw `Document` list `retrieve_context` returned as the second element of its
   `(content, artifact)` tuple. Each match is extended into `context_docs`, then reshaped into
   `{"source": doc.metadata.get("source", "Unknown"), "content": doc.page_content}` dicts. No
   second `vectorstore` call happens — the artifact rode along on the message the agent already
   produced.

4. **Q: The bug log for this file mentions `docs.metadata` (plural) vs. `doc.metadata`
   (singular) as an early bug. Why would `docs.metadata` fail, and what kind of error would it
   raise?**
   A: Inside the `for doc in docs:` loop, `docs` is the whole *list* of retrieved `Document`s, not
   the loop variable — a list has no `.metadata` attribute. Referencing `docs.metadata` instead of
   `doc.metadata` raises `AttributeError: 'list' object has no attribute 'metadata'` the first
   time the tool actually runs (a NameError/AttributeError-class bug that only surfaces once real
   documents come back, not at import time).

5. **Q: Why must `OpenAIEmbeddings(model="text-embedding-3-large", dimensions=1024)` in
   `backend/core.py` exactly match the embeddings config used in `ingestion.py`?**
   A: A query has to be embedded into the *same* vector space the stored document vectors live
   in, or cosine-similarity comparisons are meaningless. Beyond model choice, `dimensions=1024` is
   load-bearing here too: `langchain-doc-index` is a Pinecone index with a fixed 1024-dim
   schema, and OpenAI's v3 embedding models support Matryoshka-style truncation via `dimensions=`
   — mismatched dimensions would either fail outright (`PineconeApiException: Vector dimension
   ... does not match`) or, if somehow accepted, produce meaningless nearest-neighbor results.

6. **Q: What's a known limitation of `retrieve_context`/`run_llm` that's explicitly called out as
   unresolved (see README Notes for Tomorrow), and why does it happen?**
   A: Context deduplication — a single `retrieve_context` call returns up to `k=4` chunks, which
   can (and in a real smoke test, did) all share the same `source` URL, since `search_kwargs=
   {"k": 4}` retrieves the top-4 closest *chunks*, not top-4 distinct *documents*. Nothing in
   `run_llm` currently groups or dedupes by `source` before returning `context`, so a caller can
   receive several context entries that all cite the same page.

---

## Streamlit Integration (`app.py`)

**Concept.** `app.py` is a thin chat UI over `backend/core.py`'s `run_llm` — the first script in
the repo with a real frontend. `st.chat_input(...)` captures a prompt; on submission,
`run_llm(query=prompt)` is called directly (no HTTP layer — the Streamlit process imports
`backend.core` and calls the function in-process), and `sources = {doc["source"] for doc in
response["context"]}` deduplicates source URLs into a set before `create_sources_string(sources)`
formats them as a numbered markdown list appended to the answer. Chat history persists across
reruns via `st.session_state["user_prompt_history"]` / `st.session_state["chat_answers_history"]`,
which Streamlit's execution model requires explicitly — the whole script reruns top-to-bottom on
every interaction, so anything that must survive a rerun has to live in `st.session_state`, not a
local variable.

### Interview Questions

1. **Q: Why does `app.py` need `st.session_state` at all, instead of just keeping
   `chat_answers_history` as a normal Python list at module scope?**
   A: Streamlit reruns the *entire script* from top to bottom on every user interaction (like
   submitting a new chat prompt) — a plain module-level list would be reinitialized to empty on
   every rerun, wiping history. `st.session_state` is a dict that Streamlit preserves across
   reruns for the same browser session, which is why history only gets initialized once, guarded
   by `if "chat_answers_history" not in st.session_state:`.

2. **Q: Where does `create_sources_string`'s deduplication actually happen, and why does it
   matter given `backend/core.py`'s current known limitation?**
   A: The dedup happens one line earlier, in `app.py` itself: `sources = {doc["source"] for doc in
   response["context"]}` builds a Python `set`, which collapses duplicate URLs before
   `create_sources_string` ever sees them. This matters because `run_llm`'s `context` list can
   contain multiple chunks citing the same `source` (see the retrieval track's known limitation
   above) — `app.py` papers over that at the UI layer by deduping on display, rather than
   `backend/core.py` deduping at the source.

3. **Q: `app.py` calls `run_llm` directly rather than over HTTP. What's the tradeoff of doing it
   this way versus a FastAPI/Flask backend the README's Notes for Tomorrow mentions adding?**
   A: Calling `run_llm` in-process is simpler — no server to run, no network layer, no request/
   response serialization to design — which is exactly why it's the fastest way to get a working
   UI. The tradeoff is coupling: the Streamlit process *is* the only client of `run_llm`; nothing
   else (a mobile app, another service, a CLI) can reuse the RAG agent without going through
   Streamlit too. Exposing `run_llm` over an HTTP endpoint would decouple "the RAG logic" from
   "this one UI."

4. **Q: Why is `formatted_response` (answer + sources string) stored in
   `chat_answers_history` rather than storing the raw `response` dict and formatting at render
   time?**
   A: It's a simplicity/coupling tradeoff, not a correctness requirement — formatting once at
   append-time means the render loop (`for i, (user_query, generated_response) in enumerate(...)`)
   just prints two flat strings per turn with no reformatting logic. The cost is that if the
   sources-formatting logic changes later, old history entries already baked into
   `chat_answers_history` won't reflect it — only new turns would.
