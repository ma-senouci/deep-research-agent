import os
import re
import asyncio
import gradio as gr
from dotenv import load_dotenv
from logic_agents import run_pipeline

load_dotenv(override=True)


def _sanitize_error(msg: str) -> str:
    if any(k in msg.lower() for k in ("rate limit", "ratelimit", "429")):
        return "Service temporarily busy, please try again."
    msg = re.sub(r"^\w+(\.\w+)*Error:\s*", "", msg)
    msg = re.sub(r'File ".+?", line \d+.*', "", msg)
    msg = re.sub(r"Traceback.*", "", msg, flags=re.DOTALL)
    return msg.strip()[:200] or "An unexpected error occurred."


async def handle_research(query, email, openai_key, resend_key):
    if not query or not query.strip():
        yield "❌ Please enter a research query.", "", ""
        return

    if openai_key and openai_key.strip():
        os.environ["OPENAI_API_KEY"] = openai_key.strip()
    if resend_key and resend_key.strip():
        os.environ["RESEND_API_KEY"] = resend_key.strip()

    if not os.getenv("OPENAI_API_KEY"):
        yield "🔑 OpenAI API key is required. Please provide it in the API Keys section or set OPENAI_API_KEY in .env.", "", ""
        return

    queue, statuses = asyncio.Queue(), []
    task = asyncio.create_task(run_pipeline(
        query=query.strip(), answers=[],
        recipient_email=email.strip() if email and email.strip() else None,
        status_callback=lambda msg: queue.put_nowait(msg),
    ))

    while not task.done():
        try:
            statuses.append(await asyncio.wait_for(queue.get(), timeout=0.2))
            yield "\n".join(statuses), "", ""
        except asyncio.TimeoutError:
            continue

    while not queue.empty():
        statuses.append(queue.get_nowait())

    try:
        result = task.result()
    except Exception as exc:
        statuses.append(f"❌ Pipeline error: {_sanitize_error(str(exc))}")
        yield "\n".join(statuses), "", ""
        return

    trace_link = f"[View Trace]({result.trace_url})" if result.trace_url else ""
    if not result.success:
        statuses.append(f"❌ Pipeline failed: {_sanitize_error(result.error)}")
        yield "\n".join(statuses), "", trace_link
        return

    rpt = result.data
    md = f"# {rpt.title}\n\n*{rpt.overview}*\n\n{rpt.body}\n\n"
    md += "## Follow-up Questions\n\n"
    md += "\n".join(f"- {q}" for q in rpt.follow_up_questions)
    statuses.append("✅ Done")
    yield "\n".join(statuses), md, trace_link


with gr.Blocks(title="DeepResearch") as ui:
    gr.Markdown("# 🔬 DeepResearch\nAI-powered multi-agent research assistant")
    query = gr.Textbox(
        label="🔍 Research Query",
        placeholder="What would you like to research?",
        lines=3,
    )
    email = gr.Textbox(
        label="📧 Recipient Email (optional)",
        placeholder="you@example.com",
    )

    with gr.Accordion("🔑 API Keys", open=False):
        gr.Markdown("Keys entered here override `.env`.")
        openai_key = gr.Textbox(label="OpenAI API Key (required unless set in .env)", type="password")
        resend_key = gr.Textbox(label="Resend API Key (optional, email only)", type="password")

    btn = gr.Button("🚀 Research", variant="primary")
    status = gr.Textbox(label="📡 Status", interactive=False)
    report_out = gr.Markdown(label="📄 Report")
    trace_out = gr.Markdown(label="🔗 Trace")

    btn.click(
        fn=handle_research,
        inputs=[query, email, openai_key, resend_key],
        outputs=[status, report_out, trace_out],
    )

if __name__ == "__main__":
    ui.queue()
    ui.launch()
