from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# A small knowledge base to index. In a real app this would come from
# document loaders (PDFs, web pages, a database, etc.) instead of a
# hardcoded string.
RAW_TEXT = """
LangChain Expression Language (LCEL) is a declarative way to compose chains
of Runnables using the | (pipe) operator. Each step in an LCEL chain takes
the output of the previous step as its input, and the whole chain itself
is a Runnable, so chains can be nested inside other chains.

Retrieval-Augmented Generation (RAG) is a pattern where a language model's
answer is grounded in documents retrieved at query time, rather than
relying only on what the model memorized during training. A retriever
searches a vector store for chunks similar to the question, and those
chunks are inserted into the prompt as context before the model answers.

A vector store holds documents as embeddings, which are numeric vectors
that capture semantic meaning. To search a vector store, the incoming
query is embedded with the same embedding model, and the store returns
the documents whose vectors are closest to the query vector, typically
using cosine similarity.

Text splitters break long documents into smaller chunks before they are
embedded and stored. Chunking matters because embedding models have
limited context windows, and smaller, focused chunks tend to produce
more precise retrieval than embedding an entire long document at once.

Agents differ from simple chains in that an agent uses a language model
to decide which tool to call and in what order, based on the input and
intermediate results, rather than following a fixed sequence of steps
defined in advance.
"""


def build_vectorstore() -> InMemoryVectorStore:
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    chunks = splitter.split_text(RAW_TEXT)
    documents = [Document(page_content=chunk) for chunk in chunks]

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return InMemoryVectorStore.from_documents(documents, embeddings)


def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def main():
    vectorstore = build_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    prompt = ChatPromptTemplate.from_template(
        "Answer the question using only the context below. "
        "If the answer isn't in the context, say you don't know.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}"
    )
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    parser = StrOutputParser()

    rag_chain = (
        RunnableParallel(
            context=retriever | RunnableLambda(format_docs),
            question=RunnablePassthrough(),
        )
        | prompt
        | model
        | parser
    )

    question = "How does chunking relate to retrieval quality?"
    print(f"Question: {question}\n")
    print("Answer:", rag_chain.invoke(question))


if __name__ == "__main__":
    main()
