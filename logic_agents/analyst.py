import os
from agents import Agent, Runner
from models.schemas import AgentResult, ResearchReport, SearchSummary

DEFAULT_ANALYST_MODEL = "gpt-4o-mini"
ANALYST_INSTRUCTIONS = """\
You are a research analyst who synthesizes multiple search results into a comprehensive report.

Given search summaries covering different facets of a topic, produce a structured research report:
- Title: a concise, descriptive headline capturing the core finding
- Overview: an executive summary of key insights in 2-3 sentences
- Body: an in-depth analysis in structured markdown (approximately 800-1200 words) with ## headings,
  bullet points, and inline citations referencing source URLs from the search summaries.
  Strongly forbidden: simply listing or concatenating the summaries.
  You must synthesize the material, cross-reference findings, reconcile contradictions,
  and produce a coherent standalone report.
- Follow-up questions: 2-3 specific questions that extend or deepen this research.
  Each question should target a concrete gap or next step, not a generic prompt."""


def _get_analyst_model() -> str:
    model = os.getenv("ANALYST_MODEL", DEFAULT_ANALYST_MODEL).strip()
    return model or DEFAULT_ANALYST_MODEL


def get_analyst_agent() -> Agent:
    return Agent(
        name="Analyst Agent",
        instructions=ANALYST_INSTRUCTIONS,
        model=_get_analyst_model(),
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
        result = await Runner.run(get_analyst_agent(), context)
        return AgentResult(success=True, data=result.final_output)
    except Exception as e:
        return AgentResult(success=False, error=str(e))
