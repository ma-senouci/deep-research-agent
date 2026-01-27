from agents import Agent, Runner
from models.schemas import AgentResult, ResearchReport, SearchSummary

analyst_agent = Agent(
    name="Analyst Agent",
    instructions="""\
You are a research analyst who synthesizes multiple search results into a comprehensive report.

Given search summaries covering different facets of a topic, produce a structured research report:
- Title: a concise, descriptive headline capturing the core finding
- Overview: an executive summary of key insights in 2-3 sentences
- Body: an in-depth analysis in structured markdown (800-1500 words) with ## headings,
  bullet points, and inline citations referencing source URLs from the search summaries.
  Synthesize and cross-reference findings — do NOT simply concatenate summaries.
- Follow-up questions: 2-3 specific questions that extend or deepen this research.
  Each question should target a concrete gap or next step, not a generic prompt.""",
    model="gpt-4o-mini",
    output_type=ResearchReport,
)


async def run_analyst(summaries: list[SearchSummary]) -> AgentResult[ResearchReport]:
    if not summaries:
        return AgentResult(success=False, error="No search summaries provided")
    try:
        context = "\n\n".join(
            f"### {s.query}\n{s.summary}\nSources: {', '.join(s.sources)}"
            for s in summaries
        )
        result = await Runner.run(analyst_agent, context)
        return AgentResult(success=True, data=result.final_output)
    except Exception as e:
        return AgentResult(success=False, error=str(e))
