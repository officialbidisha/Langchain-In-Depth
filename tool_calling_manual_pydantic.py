# Manual tool-calling with a Pydantic-schema tool (no @tool decorator).
#
# Binding a plain Pydantic model gives the LLM a schema to request calls
# against, but there's no BaseTool object — nothing auto-executes and
# nothing auto-wraps the result into a ToolMessage. Both are written by hand.

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from tavily import TavilyClient
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, ToolMessage

load_dotenv()
tavily_client = TavilyClient()


# Schema only — the docstring becomes the tool description the model sees,
# same job the @tool docstring does in teach_tool_calling.py.
class GetNewJobs(BaseModel):
    """Search for new remote job postings."""
    query: str = Field(description="The search query for job postings.")
    search_depth: str = Field(default="advanced", description="The depth of the search, e.g., 'basic' or 'advanced'.")
    max_results: int = Field(default=15, description="The maximum number of job postings to return.")


# Implementation, kept separate from the schema above — unlike @tool, which
# fuses schema + implementation into one BaseTool object.
def get_new_jobs(query, search_depth="advanced", max_results=15):
    print(f"\nSearching for:\n{query}\n")
    return tavily_client.search(query=query, search_depth=search_depth, max_results=max_results)


if __name__ == "__main__":
    model = ChatOpenAI(model="gpt-4o", temperature=0)
    binded_model = model.bind_tools([GetNewJobs])  # pass the Pydantic CLASS itself

    messages = [HumanMessage(content="Find new software engineering remote jobs in India that are top MNCs of level Meta, Google, Salesforce, Uber")]
    result = binded_model.invoke(messages)
    print(result.tool_calls)
    messages.append(result)  # keep the AIMessage(tool_calls=...) in history

    for tool_call in result.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        if tool_name == "GetNewJobs":
            # No .invoke() here — no BaseTool exists, so args are unpacked
            # by hand and the ToolMessage is built manually.
            tool_output = get_new_jobs(**tool_args)
            tool_message = ToolMessage(content=str(tool_output), tool_call_id=tool_call["id"])
            messages.append(tool_message)

    final_result = binded_model.invoke(messages)
    print(final_result.content)
