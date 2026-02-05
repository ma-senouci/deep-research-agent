import os
import gradio as gr
from dotenv import load_dotenv
from logic_agents import run_pipeline

load_dotenv(override=True)


async def handle_research(query, email, openai_key, resend_key):
    if not query or not query.strip():
        return "❌ Please enter a research query.", ""

    if openai_key and openai_key.strip():
        os.environ["OPENAI_API_KEY"] = openai_key.strip()
    if resend_key and resend_key.strip():
        os.environ["RESEND_API_KEY"] = resend_key.strip()

    result = await run_pipeline(
        query=query.strip(),
        answers=[],
        recipient_email=email.strip() if email and email.strip() else None,
    )

    if not result.success:
        return f"❌ Pipeline failed: {result.error}", ""

    report = result.data
    md = f"# {report.title}\n\n"
    md += f"*{report.overview}*\n\n"
    md += report.body + "\n\n"
    md += "## Follow-up Questions\n\n"
    md += "\n".join(f"- {q}" for q in report.follow_up_questions)
    return "✅ Done", md


with gr.Blocks(title="DeepResearch") as ui:
    gr.Markdown("# 🔬 DeepResearch\nAI-powered multi-agent research assistant")

    query = gr.Textbox(
        label="Research Query",
        placeholder="What would you like to research?",
        lines=3,
    )
    email = gr.Textbox(label="Recipient Email (optional)", placeholder="you@example.com")

    with gr.Accordion("API Keys (optional — overrides .env)", open=False):
        openai_key = gr.Textbox(label="OpenAI API Key", type="password")
        resend_key = gr.Textbox(label="Resend API Key", type="password")

    btn = gr.Button("🚀 Research", variant="primary")
    status = gr.Textbox(label="Status", interactive=False)
    report_out = gr.Markdown(label="Report")

    btn.click(
        fn=handle_research,
        inputs=[query, email, openai_key, resend_key],
        outputs=[status, report_out],
    )

if __name__ == "__main__":
    ui.launch()
