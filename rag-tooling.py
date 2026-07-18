from dotenv import load_dotenv
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

load_dotenv()

# ---------------------------------------------------------
# STEP 0: A fake "document" object, just to mimic what a
# real retriever (e.g. from a vector store) would return.
# ---------------------------------------------------------
class Document:
    def __init__(self, page_content: str):
        self.page_content = page_content


# ---------------------------------------------------------
# STEP 1: A mock retriever.
# In a real app, this would be something like:
#   retriever = vectorstore.as_retriever()
# and it would search embeddings to find relevant chunks.
# Here we just hardcode two "matching documents" for demo.
# ---------------------------------------------------------
class FakeRetriever:
    def invoke(self, question: str):
        print(f"[Retriever] Searching docs for: {question!r}")
        return [
            Document("LCEL stands for LangChain Expression Language."),
            Document("It lets you compose chains using the | (pipe) operator."),
        ]

retriever = FakeRetriever()


# ---------------------------------------------------------
# STEP 2: "Assistant A"'s second job — squish the list of
# Document objects into one single string of context text.
# This is a plain Python function, so we wrap it in
# RunnableLambda to make it chainable with |.
# ---------------------------------------------------------
def format_docs(docs: list[Document]) -> str:
    print(f"[format_docs] Formatting {len(docs)} documents into one string")
    return "\n\n".join(d.page_content for d in docs)


# ---------------------------------------------------------
# STEP 3: Build the RAG chain.
#
# RunnableParallel runs BOTH branches below on the SAME
# input (the raw question string) at the same time:
#
#   "context"  branch -> Assistant A: search + summarize
#   "question" branch -> Assistant B: just pass it through
#
# The result is a dict:
#   {"context": "<formatted docs>", "question": "<original question>"}
# ---------------------------------------------------------
retrieval_step = RunnableParallel(
    context=RunnableLambda(retriever.invoke) | RunnableLambda(format_docs),  # Assistant A: transforms input
    question=RunnablePassthrough(),                    # Assistant B: preserves input untouched
)


# ---------------------------------------------------------
# STEP 4: The prompt template.
# Its {context} and {question} placeholders must match the
# dict keys produced by RunnableParallel above EXACTLY,
# or you'll get a KeyError when this step runs.
# ---------------------------------------------------------
prompt = ChatPromptTemplate.from_template(
    "Answer the question using only this context:\n"
    "{context}\n\n"
    "Question: {question}"
)


# ---------------------------------------------------------
# STEP 5: The model that generates the final answer.
# ---------------------------------------------------------
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ---------------------------------------------------------
# STEP 6: Output parser — converts the model's AIMessage
# object into a plain string, so callers don't need to
# know about LangChain message types at all.
# ---------------------------------------------------------
parser = StrOutputParser()


# ---------------------------------------------------------
# STEP 7: Wire it all together with the pipe operator.
# Data flows top to bottom:
#   raw question
#     -> retrieval_step   (produces {"context": ..., "question": ...})
#     -> prompt            (fills template, produces a PromptValue)
#     -> model              (produces an AIMessage)
#     -> parser              (produces a plain string)
# ---------------------------------------------------------
rag_chain = retrieval_step | prompt | model | parser


# ---------------------------------------------------------
# STEP 8: Run it.
# ---------------------------------------------------------
if __name__ == "__main__":
    answer = rag_chain.invoke("What is LCEL?")
    print("\n[Final Answer]:", answer)