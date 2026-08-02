from dotenv import load_dotenv
from typing import Any, Dict
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import ToolMessage
from langchain.tools import tool
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings

load_dotenv()

#initialise embeddings
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    dimensions=1024,  # must match the langchain-doc-index Pinecone index (dimension=1024)
    show_progress_bar=True,
    chunk_size=50, 
    retry_min_seconds=10,
)

#initialise vectorstore
vectorstore = PineconeVectorStore(index_name="langchain-doc-index", embedding=embeddings)

# chat client to read the LLM request
# generic function that takes a string and generates a chat model
model = init_chat_model("gpt-4o-mini", model_provider="openai", temperature=0.0, max_tokens=2000)

@tool(response_format = "content_and_artifact")
def retrieve_context(query:str):
    """Retrieve relevant context from the Pinecone vector store based on the query."""
    docs = vectorstore.as_retriever(search_kwargs={"k": 4}).invoke(query)
    parts = []

    for doc in docs:
        source = doc.metadata.get("source", "Unknown")
        parts.append(f"Source: {source}\n\nContent: {doc.page_content}")

    serialized_docs = "\n\n".join(parts)  
    return serialized_docs, docs


def run_llm(query:str) -> Dict[str, Any]:
    """Run the LLM with the provided query and return the response."""
    system_prompt = (
        "You are a helpful assistant that provides information based on the provided context. "
        "Use the context to answer the user's query accurately and concisely. "
        "If the context does not contain relevant information, respond with 'I don't know.'"
    )
    agent = create_agent(
        model=model,
        tools=[retrieve_context],
        system_prompt=system_prompt,
    )

    messages = [{"role": "user", "content": query}]
    response = agent.invoke({"messages": messages})

    answer = response["messages"][-1].content

    # retrieve_context returns (content, artifact) — the artifact (raw Documents)
    # is attached to the ToolMessage the agent produced when it called the tool.
    context_docs = []
    for msg in response["messages"]:
        if isinstance(msg, ToolMessage) and msg.name == "retrieve_context" and msg.artifact:
            context_docs.extend(msg.artifact)

    context = [
        {"source": doc.metadata.get("source", "Unknown"), "content": doc.page_content}
        for doc in context_docs
    ]

    return {"answer": answer, "context": context}


if __name__ == "__main__":
    user_query = "What is LangChain?"
    result = run_llm(user_query)
    print(result["answer"])
    print(f"\n--- {len(result['context'])} source(s) ---")
    for doc in result["context"]:
        print(f"- {doc['source']}")