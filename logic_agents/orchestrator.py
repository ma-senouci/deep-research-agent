from collections.abc import Callable
from agents import Agent, trace
from models.schemas import AgentResult, ResearchReport
from logic_agents.strategist import run_strategist
from logic_agents.scout import run_scouts
from logic_agents.analyst import run_analyst
from logic_agents.delivery import run_delivery

TRACE_BASE = "https://platform.openai.com/traces/"

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

    trace_url = None
    try:
        with trace("DeepResearch Pipeline") as pipeline_trace:
            trace_url = TRACE_BASE + pipeline_trace.trace_id

            _update("Planning...")
            strategist_result = await run_strategist(query, answers)
            if not strategist_result.success:
                return AgentResult(
                    success=False, error=f"Strategist failed: {strategist_result.error}",
                    trace_url=trace_url,
                )

            queries = [t.query for t in strategist_result.data.searches]
            _update(f"Searching ({len(queries)} queries)...")
            scout_results = await run_scouts(queries, status_callback=status_callback)
            summaries = [r.data for r in scout_results if r.success and r.data]
            if not summaries:
                _update("❌ All searches failed")
                return AgentResult(
                    success=False,
                    error="No search results available — please try rephrasing your query.",
                    trace_url=trace_url,
                )
            failed = len(scout_results) - len(summaries)
            if failed:
                _update(f"⚠️ {failed}/{len(scout_results)} searches failed, continuing with {len(summaries)} results")

            _update("Synthesizing...")
            analyst_result = await run_analyst(summaries)
            if not analyst_result.success:
                return AgentResult(
                    success=False, error=f"Analyst failed: {analyst_result.error}",
                    trace_url=trace_url,
                )

            report = analyst_result.data
            if recipient_email:
                _update("Sending...")
                delivery_result = await run_delivery(report, recipient_email)
                if not delivery_result.success:
                    _update(f"⚠️ Email delivery failed: {delivery_result.error}")

            return AgentResult(success=True, data=report, trace_url=trace_url)
    except Exception as e:
        return AgentResult(success=False, error=str(e), trace_url=trace_url)
