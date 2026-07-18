import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from tavily import TavilyClient

load_dotenv()

tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


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


SYSTEM_PROMPT = """
You are a job-search agent.

Follow these rules:

1. Search using the complete role, skills, and location.
2. From the search results, identify every URL that looks like
   an individual job posting (not a job-board search page,
   category listing, or company careers landing page).
3. Ignore job-board landing pages and search-result pages.
4. Call get_job_details on EVERY candidate individual job
   posting URL, one at a time, not just the first one you find.
   Keep working through the remaining candidates until you have
   verified five jobs or have checked all of them.
5. A job is verified only if ALL of the following are literally
   true in the extracted content — never infer, guess, or use
   hedging language like "may include" or "likely uses":
   a. The text contains an explicit, literal mention of
      "LangChain", "LangGraph", or "LangSmith". Generic mentions
      of "LLMs", "RAG", "AI agents", or "Generative AI" do NOT
      count as evidence, even if the role sounds related.
   b. The text explicitly states the role is remote. A role that
      is on-site, hybrid, or doesn't mention remote status at all
      is NOT verified — do not write "Yes (but listed on-site)"
      or similar contradictions; if it isn't clearly remote, drop
      the job entirely.
   c. The text explicitly confirms candidates in India are
      eligible — either by stating an India location/region, or
      by stating the role is open with no location/country
      restriction. A remote role restricted to another country
      (e.g. "Remote (US-based)") is NOT verified.
6. Quality and strict rule-following matter more than reaching
   five results. Excluding a job is always safer than including
   one that fails (a), (b), or (c) above.
7. Do not invent URLs or job details.
8. If, after checking every reasonable candidate from the first
   search, you have fewer than five verified jobs, call get_jobs
   again with a different query (e.g. adding "site:linkedin.com/jobs",
   "site:indeed.com", or "site:naukri.com") to surface new candidates,
   then repeat steps 2-6. Do this until you reach five verified jobs
   or two additional searches have found no new verified jobs.
9. Return at most five verified jobs.
10. It is expected and fine to return fewer than five jobs, or
    zero, if that's all that genuinely passes every rule above.

For every job, provide:

- Role
- Company
- Location
- Remote eligibility
- LangChain, LangGraph, or LangSmith evidence
- Required experience
- Key responsibilities
- Application URL
"""


model = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
)

agent = create_agent(
    model=model,
    tools=[get_jobs, get_job_details],
    system_prompt=SYSTEM_PROMPT,
)


result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": """
                Find up to five remote AI Engineer jobs available
                to candidates in India.

                The job description must explicitly mention
                LangChain, LangGraph, or LangSmith.
                """,
            }
        ]
    }
)


print("\nFULL AGENT TRACE\n")

for message in result["messages"]:
    message.pretty_print()


print("\nFINAL ANSWER\n")
print(result["messages"][-1].content)