# 🦜 LangChain-In-Depth

A comprehensive learning and reference repository for **LangChain**, demonstrating practical patterns for building AI applications with retrieval-augmented generation (RAG), tool-calling agents, and LCEL chains.

> **Perfect for**: Developers learning LangChain, building production RAG pipelines, and exploring LLM-powered application architectures.

---

## 📚 Overview

This repository provides **working, production-ready examples** of core LangChain concepts, progressing from fundamentals to advanced patterns:

- **Chat Models & Prompts** – Basic LLM interactions and conversation management
- **LCEL Chains** – LangChain Expression Language for composable, reusable pipelines
- **Retrieval-Augmented Generation (RAG)** – From mock retrievers to production embeddings + vector stores
- **Tool Calling & Agents** – Building intelligent agents with tool integration using `create_agent`
- **Multi-turn Conversations** – State management and context preservation

Each example is **standalone, runnable, and well-documented** for quick learning and reference.

---

## 🎯 Key Features

✅ **Incremental Learning** – Examples build from simple to complex  
✅ **Production-Ready Code** – Real patterns used in production LLM apps  
✅ **Best Practices** – Error handling, prompt engineering, and optimization  
✅ **Easy Setup** – One command to get started with `uv`  
✅ **Clear Documentation** – Inline comments and structured examples  

---

## 📋 Requirements

- **Python 3.10+**
- **[uv](https://docs.astral.sh/uv/)** – Fast Python package manager (replaces pip + venv)
- **[OpenAI API key](https://platform.openai.com/api-keys)** – For LLM interactions
- **[Tavily API key](https://tavily.com/)** – For web search tool (agent examples)

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/officialbidisha/Langchain-In-Depth.git
cd Langchain-In-Depth
```

### 2. Install Dependencies

```bash
uv sync
```

This installs all dependencies listed in `pyproject.toml` and locks them in `uv.lock`.

### 3. Configure API Keys

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=sk-your-openai-key-here
TAVILY_API_KEY=your-tavily-api-key-here
```

> ⚠️ **Never commit `.env`** – it's already in `.gitignore`

### 4. Run Examples

Each script runs independently:

```bash
# Chat models, prompt templates, and basic LCEL chains
uv run python main.py

# RAG pipeline with a mocked retriever (no external calls)
uv run python rag-tooling.py

# Real RAG with embeddings and vector search
uv run python real_rag.py

# Tool-calling agent with Tavily web search
uv run python tool_calling.py
```

---

## 📁 Project Structure

```
Langchain-In-Depth/
├── main.py                      # Chat models, prompts, LCEL basics
├── rag-tooling.py              # RAG architecture with fake retriever
├── real_rag.py                 # Production RAG: embeddings + vector store
├── tool_calling.py             # Agent with tool integration
├── pyproject.toml              # Dependencies and project metadata
├── uv.lock                     # Locked dependency versions
├── .env                        # Local secrets (not committed)
├── .gitignore                  # Git ignore rules
├── README.md                   # This file
└── LICENSE                     # Repository license
```

---

## 🔍 Example Breakdown

### `main.py` – Foundations
- Initialize chat models (OpenAI)
- Create prompt templates
- Build LCEL chains (`|` operator)
- Handle multi-turn conversations
- Parse structured outputs

### `rag-tooling.py` – RAG Architecture
- Understand RAG pipeline shape
- Implement retrieval chains
- Use `RunnableParallel` for parallel execution
- Mock retriever for testing

### `real_rag.py` – Production RAG
- Split documents into chunks
- Generate embeddings
- Store in vector database
- Real semantic search
- End-to-end retrieval flow

### `tool_calling.py` – Agents
- Define custom tools
- Create an agent with `create_agent`
- Implement tool-calling loop
- Handle function results
- Build autonomous workflows

---

## 🛠️ Technology Stack

| Component | Purpose |
|-----------|---------|
| **LangChain** | LLM orchestration framework |
| **OpenAI API** | Language model (GPT-4, GPT-3.5) |
| **uv** | Python package management |
| **python-dotenv** | Environment variable management |
| **Tavily** | Web search tool for agents |

---

## 📖 Learning Path

1. **Start here:** `main.py` – Understand basics
2. **Next:** `rag-tooling.py` – Learn RAG architecture
3. **Then:** `real_rag.py` – Build production systems
4. **Advanced:** `tool_calling.py` – Create intelligent agents

---

## 💡 Common Workflows

### Running a Single Example with Arguments

```bash
uv run python main.py --model gpt-4
```

### Debugging with Print Statements

```bash
# Enable verbose logging
export LANGCHAIN_DEBUG=true
uv run python main.py
```

### Installing Additional Dependencies

```bash
uv pip install package-name
```

---

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError` | Run `uv sync` to ensure all dependencies are installed |
| `API key not found` | Check `.env` file exists with correct keys |
| `Connection timeout` | Verify OpenAI/Tavily services are accessible |
| `Permission denied` | Ensure API keys have correct permissions/quota |

---

## 🤝 Contributing

Contributions are welcome! To improve this repository:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/awesome-example`)
3. Commit your changes (`git commit -m "Add awesome example"`)
4. Push to the branch (`git push origin feature/awesome-example`)
5. Open a Pull Request

---

## 📝 License

This project is distributed under the terms of the [LICENSE](LICENSE) file included in this repository.

---

## 🔗 Resources

- [LangChain Documentation](https://python.langchain.com/)
- [LCEL Guide](https://python.langchain.com/docs/expression_language/)
- [RAG Best Practices](https://python.langchain.com/docs/use_cases/question_answering/)
- [Agents Documentation](https://python.langchain.com/docs/modules/agents/)
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference)

---

## ⭐ Show Your Support

If this repository helped you learn LangChain, please consider:
- ⭐ Starring the repository
- 🐦 Sharing it with others
- 💬 Providing feedback or suggestions

**Happy learning! 🚀**
