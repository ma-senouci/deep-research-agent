import asyncio
import pytest
import os
from unittest.mock import AsyncMock
from models.schemas import AgentResult, ResearchReport


REPORT = AgentResult(
    success=True,
    data=ResearchReport(
        title="Test Report",
        overview="Test overview.",
        body="## Section\n\nTest body content.",
        follow_up_questions=["Follow-up 1?", "Follow-up 2?"],
    ),
    trace_url="https://platform.openai.com/traces/test-trace-id",
)

FAILURE = AgentResult(success=False, error="Something went wrong", trace_url="https://platform.openai.com/traces/fail-id")


async def _collect(gen):
    results = []
    async for item in gen:
        results.append(item)
    return results


@pytest.mark.asyncio
async def test_handle_research_streams_status(monkeypatch):
    import app as app_mod

    async def mock_pipeline(query, answers, recipient_email=None, status_callback=None):
        if status_callback:
            status_callback("Planning...")
            await asyncio.sleep(0)
            status_callback("Searching (3 queries)...")
            await asyncio.sleep(0)
            status_callback("Synthesizing...")
        return REPORT

    monkeypatch.setattr(app_mod, "run_pipeline", mock_pipeline)
    results = await _collect(app_mod.handle_research("AI query", "", "", ""))
    statuses = [r[0] for r in results]
    assert any("Planning" in s for s in statuses)
    assert "Done" in statuses[-1]
    assert "Test Report" in results[-1][1]


@pytest.mark.asyncio
async def test_handle_research_failure(monkeypatch):
    import app as app_mod

    monkeypatch.setattr(app_mod, "run_pipeline", AsyncMock(return_value=FAILURE))
    results = await _collect(app_mod.handle_research("AI query", "", "", ""))
    final_status = results[-1][0]
    assert "failed" in final_status.lower()


@pytest.mark.asyncio
async def test_handle_research_empty_query():
    import app as app_mod

    results = await _collect(app_mod.handle_research("", "", "", ""))
    assert len(results) == 1
    assert "Please enter" in results[0][0]


@pytest.mark.asyncio
async def test_handle_research_sets_api_keys(monkeypatch):
    import app as app_mod

    monkeypatch.setattr(app_mod, "run_pipeline", AsyncMock(return_value=REPORT))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("RESEND_API_KEY", raising=False)
    await _collect(app_mod.handle_research("query", "", "sk-test-key", "re-test-key"))
    assert os.getenv("OPENAI_API_KEY") == "sk-test-key"
    assert os.getenv("RESEND_API_KEY") == "re-test-key"


@pytest.mark.asyncio
async def test_handle_research_trace_url(monkeypatch):
    import app as app_mod

    monkeypatch.setattr(app_mod, "run_pipeline", AsyncMock(return_value=REPORT))
    results = await _collect(app_mod.handle_research("AI query", "", "", ""))
    trace_md = results[-1][2]
    assert "View Trace" in trace_md
    assert "test-trace-id" in trace_md


@pytest.mark.asyncio
async def test_handle_research_pipeline_exception(monkeypatch):
    import app as app_mod

    async def exploding_pipeline(**kwargs):
        raise RuntimeError("Unexpected boom")

    monkeypatch.setattr(app_mod, "run_pipeline", exploding_pipeline)
    results = await _collect(app_mod.handle_research("AI query", "", "", ""))
    final_status = results[-1][0]
    assert "Pipeline error" in final_status
    assert "boom" in final_status.lower()
