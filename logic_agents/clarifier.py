from agents import Agent, Runner
from models.schemas import AgentResult, ClarificationResult

clarifier_agent = Agent(
    name="Clarifier Agent",
    instructions="""\
You are an expert research consultant.
Your task is to refine the user's research query by asking exactly 3 targeted clarifying questions.
The questions should help uncover missing context, define scope, set the desired depth, or clarify ambiguous terms.
Ask specific, non-overlapping questions tailored to the topic, and avoid generic or repetitive wording.
If the query is vague, ask questions that will make the research plan more focused and useful.
Output only a ClarificationResult with exactly 3 questions.""",
    model="gpt-4o-mini",
    output_type=ClarificationResult,
)


async def run_clarifier(query: str) -> AgentResult[ClarificationResult]:
    if not query or not query.strip():
        return AgentResult(success=False, error="Query must not be empty")
    try:
        result = await Runner.run(clarifier_agent, query)
        return AgentResult(success=True, data=result.final_output)
    except Exception as e:
        return AgentResult(success=False, error=str(e))
