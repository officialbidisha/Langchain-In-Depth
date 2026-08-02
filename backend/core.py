from dotenv import load_dotenv
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

model = init_chat_model("gpt-4o-mini", model_provider="openai", temperature=0.0, max_tokens=2000)

@tool(response_format = "content_and_artifact")
def retrieve_context(query:str):
    """Retrieve relevant context from the Pinecone vector store based on the query."""


