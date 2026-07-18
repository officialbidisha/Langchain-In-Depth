from typing import List

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from tavily import TavilyClient

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI

load_dotenv()
tavily_client = TavilyClient()


# --- Structured output schema -----------------------------------------
# response_format=AgentResponse (below) tells the agent to return its
# final answer as this Pydantic model instead of free-form text.

class Job(BaseModel):
    """A single job posting found by the agent."""

    title: str = Field(description="The job title / role name")
    company: str = Field(description="The company hiring for this role")
    location: str = Field(description="Where the job is located, e.g. 'Remote - San Francisco, CA'")
    url: str = Field(description="The URL of the job posting")


class AgentResponse(BaseModel):
    """The agent's final structured answer: a list of job postings found."""

    jobs: List[Job] = Field(description="A list of job postings found")


# --- Tool ---------------------------------------------------------------

@tool
def get_new_jobs(location: str):
    """Search for new remote jobs in a given location."""
    query = f"New remote jobs in {location}"
    print(f"\nSearching for:\n{query}\n")
    return tavily_client.search(
        query=query,
        search_depth="advanced",
        max_results=15,
    )


# --- Agent ---------------------------------------------------------------

llm = ChatOpenAI(model="gpt-4o", temperature=0)
agent = create_agent(llm, tools=[get_new_jobs], response_format=AgentResponse)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "Find new software engineering remote jobs in India that are top MNCs of level Meta, Google, Salesforce, Uber"}]}
)

response = result["structured_response"]
for job in response.jobs:
    print(f"Title: {job.title}")
    print(f"Company: {job.company}")
    print(f"Location: {job.location}")
    print(f"URL: {job.url}\n")
