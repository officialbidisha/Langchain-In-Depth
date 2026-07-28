from dotenv import load_dotenv
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from tavily import TavilyClient
from langchain_core.messages import HumanMessage

load_dotenv()
tavily_client = TavilyClient()




@tool
def get_new_jobs(query: str, search_depth: str = "advanced", max_results: int = 15):
    """Search for new job postings. This function accepts the search depth, maximum results, and query as parameters.
    It returns a list of job postings that match the query.
    It calls the TAVILY API to perform the search and returns the result in a structured format.
    """
    print(f"\nSearching for:\n{query}\n")
    return tavily_client.search(
        query=query,
        search_depth=search_depth,
        max_results=max_results,
    )

TOOLS = [get_new_jobs]
TOOLS_BY_NAME = {tool.name: tool for tool in TOOLS}

if __name__ == "__main__":
    model = ChatOpenAI(model="gpt-4o", temperature=0)
    binded_model = model.bind_tools(TOOLS)
    messages = [HumanMessage(content="Find new software engineering remote jobs in India that are top MNCs of level Meta, Google, Salesforce, Uber")]
    result = binded_model.invoke(messages)
    print(result.content)
    print("---tool calls---")
    print(result.tool_calls)
    messages.append(result)
    
    for tool_call in result.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_result = TOOLS_BY_NAME[tool_name].invoke(tool_call)
        messages.append(tool_result)
    res = binded_model.invoke(messages)
    print(res.content)
    
