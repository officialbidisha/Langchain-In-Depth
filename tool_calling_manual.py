import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from tavily import TavilyClient

load_dotenv()

tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


# --- Tools ----------------------------------------------------------------
# Same two tools as tool_calling.py, unchanged.

@tool
def get_jobs(role: str, skills: str, location: str) -> dict:
    """Search for remote jobs using the role, required skills, and location."""

    query = f"""
    {role} remote jobs in {location}
    requiring {skills}
    """

    print(f"\nSearching for:\n{query}\n")

    return tavily.search(
        query=query,
        search_depth="advanced",
        max_results=15,
    )


@tool
def get_job_details(url: str) -> dict:
    """Read the complete details of an individual job posting."""

    return tavily.extract(
        urls=url,
        extract_depth="advanced",
        format="markdown",
    )


TOOLS = [get_jobs, get_job_details]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}


SYSTEM_PROMPT = """
You are a job-search agent.

1. Search using the complete role, skills, and location.
2. From the search results, identify individual job posting URLs
   (not job-board search pages or company careers landing pages).
3. Call get_job_details on each candidate URL to verify it.
4. A job is verified only if the extracted content explicitly
   mentions LangChain, LangGraph, or LangSmith, explicitly states
   the role is remote, and explicitly confirms India eligibility.
5. Return up to five verified jobs with role, company, location,
   and application URL. Fewer (or zero) is fine if that's all that
   genuinely passes verification.
"""


model = ChatOpenAI(model="gpt-4o", temperature=0)

# create_agent does this binding internally; here it's explicit.
model_with_tools = model.bind_tools(TOOLS)

messages = [
    SystemMessage(content=SYSTEM_PROMPT),
    HumanMessage(
        content=(
            "Find up to five remote AI Engineer jobs available to candidates "
            "in India that explicitly mention LangChain, LangGraph, or LangSmith."
        )
    ),
]


# --- The agent loop, by hand ------------------------------------------------
# This is exactly what create_agent runs internally (via LangGraph):
#   1. Call the model with the conversation so far.
#   2. If it responded with tool_calls instead of a final answer,
#      run each requested tool and feed the results back as ToolMessages.
#   3. Repeat until the model responds with no tool_calls (done),
#      or a step limit is hit (safety valve against infinite loops).

MAX_STEPS = 10

for step in range(MAX_STEPS):
    response = model_with_tools.invoke(messages)
    messages.append(response)

    if not response.tool_calls:
        break

    for call in response.tool_calls:
        print(f"[step {step}] calling {call['name']}({call['args']})")
        tool_fn = TOOLS_BY_NAME[call["name"]]
        result = tool_fn.invoke(call["args"])
        # tool_call_id links this result back to the specific tool_call
        # the model made — required so the model knows which call it answers.
        messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))
else:
    print(f"\nStopped after {MAX_STEPS} steps without a final answer.\n")


print("\nFULL MESSAGE TRACE\n")
for message in messages:
    message.pretty_print()

print("\nFINAL ANSWER\n")
print(messages[-1].content)
