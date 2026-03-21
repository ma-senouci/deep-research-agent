from agents import Agent, Runner
from models.schemas import AgentResult, SearchPlan

strategist_agent = Agent(
    name="Strategist Agent",
    instructions="""\
You are a search strategist who creates targeted web search plans.
Given a research query and the user's clarification answers, produce 3-5 optimized search queries.
Use the clarification answers when available to narrow scope, depth, timeframe, or perspective.
For each search query, provide:
- reasoning: the angle, gap, or perspective this query is meant to cover
- query: the optimized search query itself
Prioritize non-overlapping queries that cover distinct facets of the topic, such as core concepts,
comparisons, recent developments, and practical applications.""",
    model="gpt-4o-mini",
    output_type=SearchPlan,
)


async def run_strategist(query: str, answers: list[str]) -> AgentResult[SearchPlan]:
    if not query or not query.strip():
        return AgentResult(success=False, error="Query must not be empty")
    try:
        context = f"Query: {query}\nAnswers: {answers}"
        result = await Runner.run(strategist_agent, context)
        return AgentResult(success=True, data=result.final_output)
    except Exception as e:
        return AgentResult(success=False, error=str(e))
