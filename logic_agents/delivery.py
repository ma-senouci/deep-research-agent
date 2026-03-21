import os
import resend
from pydantic import BaseModel
from agents import Agent, Runner
from models.schemas import AgentResult, DeliveryResult, ResearchReport


class _EmailContent(BaseModel):
    html: str


delivery_agent = Agent(
    name="Delivery Agent",
    instructions="""\
Format the research report as a clean HTML email body.
Preserve the report structure and convert markdown into simple email-friendly HTML.
Convert section headings to <h2>, paragraphs to <p>, and bullet lists to <ul>/<li>.
Do not add explanations, wrappers, or extra commentary.
Return only the HTML body in the 'html' field.""",
    model="gpt-4o-mini",
    output_type=_EmailContent,
)


async def run_delivery(report: ResearchReport, recipient_email: str | None = None) -> AgentResult[DeliveryResult]:
    try:
        api_key = os.getenv("RESEND_API_KEY", "")
        if not api_key:
            return AgentResult(success=False, error="RESEND_API_KEY not configured")
        to_address = recipient_email or os.getenv("RECIPIENT_EMAIL", "")
        if not to_address:
            return AgentResult(success=False, error="No recipient email provided")
        resend.api_key = api_key
        prompt = (
            f"Title: {report.title}\n"
            f"Overview: {report.overview}\n"
            f"Body:\n{report.body}\n"
            f"Follow-up questions: {', '.join(report.follow_up_questions)}"
        )
        result = await Runner.run(delivery_agent, prompt)
        content: _EmailContent = result.final_output
        from_address = os.getenv("SENDER_EMAIL", "onboarding@resend.dev")
        # Send via code instead of an LLM tool call so delivery stays deterministic
        # and we do not spend tokens on a straightforward control-flow step.
        response = resend.Emails.send({
            "from": from_address,
            "to": [to_address],
            "subject": report.title,
            "html": content.html,
        })
        email_id = response.get("id", "unknown")
        return AgentResult(success=True, data=DeliveryResult(sent=True, message=f"Report sent to {to_address} (id: {email_id})"))
    except Exception as e:
        return AgentResult(success=False, error=str(e))
