# ReAct loop with a real choice of tools.
#
# STAGE 1 -- write this yourself. See instructions from Claude.
#
# teach_react_agent.py proved the loop mechanics with one tool, where the
# model never actually had to pick anything. Now there are two tools with
# different jobs, so the prompt has to describe them and the parsing has to
# read back *which one* the model chose, not just its input.

from dotenv import load_dotenv
from tavily import TavilyClient
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

load_dotenv()
tavily_client = TavilyClient()

class JobSearchTool(BaseModel):

    """Search for new remote job postings."""
    description:str = Field(description="Search for information about a company.", default="Search for new remote job postings.")
    query:str = Field(description="The search query for job postings.")    

class CompanyInfoTool(BaseModel):
    """Search for information about a company."""
    description:str = Field(description="Search for information about a company.", default="Search for information about a company.")
    query: str = Field(description="The search query for company information.")

def search_jobs(query: str) -> str:
    print(f"Searching jobs for: {query}")
    results = tavily_client.search(query=query, max_results=3)["results"]
    return "\n".join(r["title"] for r in results)

def search_company_info(query: str) -> str:
    print(f"Searching company info for: {query}")
    results = tavily_client.search(query=query, max_results=3)["results"]
    return "\n".join(f"{r['title']}: {r['content'][:200]}" for r in results)

# TODO: name -> callable dict, same pattern as TOOLS_BY_NAME in
# teach_tool_calling.py. This is what dispatch will look up into instead of
# calling search_jobs directly.
TOOLS_BY_NAME = {JobSearchTool.__name__: search_jobs, CompanyInfoTool.__name__:search_company_info}
TOOLS_DESCRIPTIONS = {
    JobSearchTool.__name__: JobSearchTool.model_fields["description"].default,
    CompanyInfoTool.__name__: CompanyInfoTool.model_fields["description"].default,
}

print(TOOLS_DESCRIPTIONS.items())

scratchpad = "" 
question = "Find a remote software engineering job , then tell me about Salesforce's culture"
tool_names = ", ".join(TOOLS_BY_NAME.keys())
tools = "\n".join(f"{name}: {description}" for name, description in TOOLS_DESCRIPTIONS.items())


# TODO: rewrite this prompt so it lists the available tools (name + a
# one-line description of what each does) instead of hardcoding a single
# action name the way teach_react_agent.py's PROMPT does. The model can only
# choose between tools it's actually told about.
#
# Keep the same skeleton shape as teach_react_agent.py's PROMPT -- it still
# needs {question} and {scratchpad} placeholders, and the model still needs
# to see Thought/Action/Action Input/Observation as the format to follow.
PROMPT = """You are a helpful assistant. You have access to the following tools:

{tools}

Use the following format:

Thought: You should always think about what to do
Action: The action to take, should be one of [{tool_names}]
Action Input: The input to the action
Observation: The result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Final Answer: The final answer to the original input question

Do not repeat an Action with the same Action Input you've already tried -- if a search didn't return what you needed, either try a different input or answer with what you have.

Question: {question}
{scratchpad}"""

model = ChatOpenAI(model="gpt-4o", temperature=0)
for step in range(10):
    prompt = PROMPT.format(question=question, scratchpad=scratchpad, tools=tools, tool_names=tool_names)
    reply = model.invoke(prompt, stop=["\nObservation:"]).content
    print(f"\n--- step {step} ---\n{reply}")

    if "Final Answer:" in reply:
        print("\n=== Answer ===")
        print(reply.split("Final Answer:")[-1].strip())
        break

    lines = reply.splitlines()
    for line in lines:
        if(line.startswith("Action Input:")):
            action_input = line.split("Action Input:")[-1].strip()
        if(line.startswith("Action:")):
            action_name = line.split("Action:")[-1].strip()
    
    result = TOOLS_BY_NAME[action_name](action_input)
    observation = result
    scratchpad += reply + f"\nObservation: {observation}\n"
