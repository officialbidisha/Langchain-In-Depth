import asyncio
import os

import certifi
from dotenv import load_dotenv
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilyCrawl


load_dotenv()

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    dimensions=1024,  # must match the langchain-doc-index Pinecone index (dimension=1024)
    show_progress_bar=True,
    chunk_size=50,
    retry_min_seconds=10,
)

vectorstore = PineconeVectorStore(index_name="langchain-doc-index", embedding=embeddings)
tavily_crawl = TavilyCrawl()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)

UPSERT_BATCH_SIZE = 100  # keep well under Pinecone's per-request limits


async def main():
    """Main async function to run the TavilyCrawl and orchestrate the entire process."""
    res = await tavily_crawl.ainvoke({
        "url": "https://docs.langchain.com/docs/",
        "max_depth": 2,
        "extract_depth": "advanced",
        "instructions": (
            "Extract all relevant information from the documentation, including "
            "code snippets, examples, and explanations. Ensure that the extracted "
            "content is comprehensive and covers all aspects of the documentation."
        ),
    })

    all_docs = res.get("results", [])
    print(f"Total documents extracted: {len(all_docs)}")

    documents = [
        Document(page_content=doc["raw_content"], metadata={"source": doc["url"]})
        for doc in all_docs
        if doc.get("raw_content")
    ]

    if not documents:
        print("No documents with content were extracted. Exiting.")
        return

    split_docs = text_splitter.split_documents(documents)
    print(f"Total chunks to embed: {len(split_docs)}")

    for i in range(0, len(split_docs), UPSERT_BATCH_SIZE):
        batch = split_docs[i : i + UPSERT_BATCH_SIZE]
        try:
            vectorstore.add_documents(batch)
            print(f"Upserted batch {i // UPSERT_BATCH_SIZE + 1} "
                  f"({len(batch)} chunks, {i + len(batch)}/{len(split_docs)} total)")
        except Exception as e:
            print(f"Failed to upsert batch starting at index {i}: {e}")

    print("Finished ingesting documents into Pinecone.")


if __name__ == "__main__":
    asyncio.run(main())