import asyncio
from collections.abc import Callable
from agents import Agent, ModelSettings, Runner, WebSearchTool
from models.schemas import AgentResult, SearchSummary

scout_agent = Agent(
    name="Scout Agent",
    instructions="""\
You are a web research scout. For the given search query, perform a focused web search
and return a concise summary of the most relevant findings.
Write the summary itself in 2-3 paragraphs and keep it under 300 words.
Capture the main points, be succinct, ignore fluff, and avoid commentary.
Include only specific URLs or citations in the sources list.
If no sources are found, return an empty sources list.""",
    model="gpt-4o-mini",
    tools=[WebSearchTool(search_context_size="low")],
    model_settings=ModelSettings(tool_choice="required"),
    output_type=SearchSummary,
)


async def run_scout(query: str) -> AgentResult[SearchSummary]:
    try:
        result = await Runner.run(scout_agent, query)
        return AgentResult(success=True, data=result.final_output)
    except Exception as e:
        return AgentResult(success=False, error=str(e))


async def run_scouts(
    queries: list[str],
    status_callback: Callable[[str], None] | None = None,
) -> list[AgentResult[SearchSummary]]:
    total = len(queries)
    results = await asyncio.gather(*[run_scout(q) for q in queries])
    for i, result in enumerate(results):
        if status_callback:
            status_callback(f"{i + 1}/{total} searches complete")
    return results
