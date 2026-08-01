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

# add the following lines to set the SSL context for requests
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()


# Chunk-size: it is used for the number of text objects to be embedded for OpenAI Embeddings.
# The default value is 50, but you can adjust it based on your needs and the size of your documents.
# A larger chunk size may speed up the embedding process, but it may also increase memory usage. 
# A smaller chunk size may reduce memory usage, but it may also slow down the embedding process.
# A high number will cause rate limiting errors, so it is recommended to keep it at 50 or lower.


embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    dimensions=1024,  # must match the langchain-doc-index Pinecone index (dimension=1024)
    show_progress_bar=True,
    chunk_size=50,
    retry_min_seconds=10
)

#vector store

vectorstore = PineconeVectorStore(index_name="langchain-doc-index", embedding=embeddings)
# scraping and crawling documentation
tavily_crawl = TavilyCrawl()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)


async def main():
    """Main async function to run the TavilyCrawl and orchestrate the entire process."""
    res = tavily_crawl.invoke({
        "url": "https://docs.langchain.com/docs/",
        "max_depth": 2,
        "extract_depth": "advanced",
        "instructions": "Extract all relevant information from the documentation, including code snippets, examples, and explanations. Ensure that the extracted content is comprehensive and covers all aspects of the documentation."
    })
    all_docs = res["results"]
    print(f"Total documents extracted: {len(all_docs)}")

    documents = [
        Document(page_content=doc["raw_content"], metadata={"source": doc["url"]})
        for doc in all_docs
        if doc.get("raw_content")
    ]
    split_docs = text_splitter.split_documents(documents)
    print(f"Total chunks to embed: {len(split_docs)}")

    vectorstore.add_documents(split_docs)
    print("Finished ingesting documents into Pinecone.")


if __name__ == "__main__":
    asyncio.run(main())