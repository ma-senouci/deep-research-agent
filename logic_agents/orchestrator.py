from collections.abc import Callable
from agents import Agent
from models.schemas import AgentResult, ResearchReport
from logic_agents.strategist import run_strategist
from logic_agents.scout import run_scouts
from logic_agents.analyst import run_analyst
from logic_agents.delivery import run_delivery

orchestrator_agent = Agent(
    name="Orchestrator Agent",
    instructions="Coordinate the full research pipeline: plan → search → analyze → deliver.",
    model="gpt-4o-mini",
    output_type=ResearchReport,
)


async def run_pipeline(
    query: str,
    answers: list[str],
    recipient_email: str | None = None,
    status_callback: Callable[[str], None] | None = None,
) -> AgentResult[ResearchReport]:
    def _update(msg: str) -> None:
        if status_callback:
            status_callback(msg)

    try:
        _update("Planning...")
        strategist_result = await run_strategist(query, answers)
        if not strategist_result.success:
            return AgentResult(success=False, error=f"Strategist failed: {strategist_result.error}")

        queries = [t.query for t in strategist_result.data.searches]
        _update(f"Searching ({len(queries)} queries)...")
        scout_results = await run_scouts(queries, status_callback=status_callback)
        summaries = [r.data for r in scout_results if r.success and r.data]

        _update("Synthesizing...")
        analyst_result = await run_analyst(summaries)
        if not analyst_result.success:
            return AgentResult(success=False, error=f"Analyst failed: {analyst_result.error}")

        report = analyst_result.data
        _update("Sending...")
        await run_delivery(report, recipient_email)

        return AgentResult(success=True, data=report)
    except Exception as e:
        return AgentResult(success=False, error=str(e))
