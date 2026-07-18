# LangChain-Depth

A collection of small, focused Python examples for learning [LangChain](https://www.langchain.com/) — from basic chat model calls and LCEL chains to retrieval-augmented generation and tool-calling agents.

## Overview

This project provides working, runnable examples of core LangChain concepts, built incrementally:

- Chat models, prompt templates, and multi-turn conversations
- LCEL (`|`) chains with `RunnableParallel` / `RunnablePassthrough` and output parsers
- Retrieval-augmented generation (RAG), from a mocked retriever up to a real embeddings + vector store pipeline
- Tool calling and agents built with `create_agent` (LangChain v1.0's agent interface)

Dependency management and locking are handled via [uv](https://docs.astral.sh/uv/).

## Requirements

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) package manager
- An [OpenAI API key](https://platform.openai.com/api-keys)
- A [Tavily API key](https://tavily.com/) (for the tool-calling / agent examples)

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/officialbidisha-a11y/LangChain-Depth.git
   cd LangChain-Depth
   ```
2. Install dependencies:
   ```bash
   uv sync
   ```
3. Create a `.env` file in the project root with your API keys:
   ```
   OPENAI_API_KEY=your-openai-key
   TAVILY_API_KEY=your-tavily-key
   ```

## Usage

Each script is a standalone example — run any of them directly:

```bash
uv run python main.py            # chat models, prompt templates, and LCEL basics
uv run python rag-tooling.py      # RAG pipeline shape using a mocked retriever
uv run python real_rag.py         # real RAG: text splitting, embeddings, vector search
uv run python tool_calling.py     # tool calling with a create_agent-based agent
```

## Project Structure

```
.
├── main.py             # Chat models, prompts, multi-turn messages, LCEL chains
├── rag-tooling.py       # RAG chain shape with a fake in-memory retriever
├── real_rag.py          # RAG with real embeddings and vector store retrieval
├── tool_calling.py      # Tool-calling agent (create_agent + a Tavily search tool)
├── pyproject.toml       # Project metadata and dependencies
├── uv.lock              # Locked dependency versions
└── .env                 # Local environment variables (not committed)
```

## License

Distributed under the terms of the [LICENSE](LICENSE) file included in this repository.
