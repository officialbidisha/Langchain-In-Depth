from dotenv import load_dotenv
load_dotenv()
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from tavily import TavilyClient
from langchain.agents import create_agent

tavily = TavilyClient()

@tool
def get_jobs(location:str) -> str:
    """Get job postings for a given location."""
    return tavily.search(location)


tools = [get_jobs]
model = ChatOpenAI(model="gpt-4o-mini", temperature=0.9, max_tokens=800)
agent = create_agent(model=model, tools=tools)
question = "search for 3 job postings for an ai engineer using langchain in the remote india"
result = agent.invoke({"messages": [{"role": "user", "content": question}]})

print(result["messages"][-1].content)