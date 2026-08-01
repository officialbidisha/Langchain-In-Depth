# A minimal ReAct agent: ask -> think -> act -> observe -> repeat -> answer.
# One tool only, so there's nothing to "choose between" yet -- just enough
# moving parts to see the loop itself.

from dotenv import load_dotenv
from tavily import TavilyClient
from langchain_openai import ChatOpenAI

load_dotenv()
tavily_client = TavilyClient()


def search_jobs(query: str) -> str:
    print(f"Searching for: {query}")
    results = tavily_client.search(query=query, max_results=3)["results"]
    return "\n".join(r["title"] for r in results)


PROMPT = """Answer the question using this format:

Thought: what you're thinking
Action: search_jobs
Action Input: what to search for
Observation: (you'll be given this)
... repeat Thought/Action/Action Input/Observation as needed ...
Thought: I now know the answer
Final Answer: the answer

Question: {question}
{scratchpad}"""

question = "Find a remote software engineering job at Meta in India"
scratchpad = ""
model = ChatOpenAI(model="gpt-4o", temperature=0)

for step in range(5):
    prompt = PROMPT.format(question=question, scratchpad=scratchpad)
    reply = model.invoke(prompt, stop=["\nObservation:"]).content
    print(f"\n--- step {step} ---\n{reply}")

    if "Final Answer:" in reply:
        print("\n=== Answer ===")
        print(reply.split("Final Answer:")[-1].strip())
        break

    action_input = reply.split("Action Input:")[-1].strip()
    observation = search_jobs(action_input)
    scratchpad += reply + f"\nObservation: {observation}\n"
