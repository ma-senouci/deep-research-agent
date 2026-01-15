from agents import Agent, Runner
from models.schemas import AgentResult, ClarificationResult

clarifier_agent = Agent(
    name="Clarifier Agent",
    instructions=(
        "You are an expert research consultant known for your curiosity. "
        "Your goal is to help users refine their research queries by asking exactly 3 targeted questions. "
        "These questions should uncover missing context, specify depth, or clarify ambiguous terms. "
        "Be specific to the topic and avoid generic questions. "
        "If the query is vague, generate focused questions to guide the user. "
        "Output ONLY a ClarificationResult with exactly 3 questions."
    ),
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
