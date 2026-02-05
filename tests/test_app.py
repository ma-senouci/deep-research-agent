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
)

FAILURE = AgentResult(success=False, error="Something went wrong")


@pytest.mark.asyncio
async def test_handle_research_success(monkeypatch):
    import app as app_mod

    monkeypatch.setattr(app_mod, "run_pipeline", AsyncMock(return_value=REPORT))
    status, md = await app_mod.handle_research("AI query", "", "", "")
    assert "Done" in status
    assert "Test Report" in md


@pytest.mark.asyncio
async def test_handle_research_failure(monkeypatch):
    import app as app_mod

    monkeypatch.setattr(app_mod, "run_pipeline", AsyncMock(return_value=FAILURE))
    status, md = await app_mod.handle_research("AI query", "", "", "")
    assert "failed" in status.lower()


@pytest.mark.asyncio
async def test_handle_research_empty_query():
    import app as app_mod

    status, md = await app_mod.handle_research("", "", "", "")
    assert "Please enter" in status


@pytest.mark.asyncio
async def test_handle_research_sets_api_keys(monkeypatch):
    import app as app_mod

    monkeypatch.setattr(app_mod, "run_pipeline", AsyncMock(return_value=REPORT))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("RESEND_API_KEY", raising=False)
    await app_mod.handle_research("query", "", "sk-test-key", "re-test-key")
    assert os.getenv("OPENAI_API_KEY") == "sk-test-key"
    assert os.getenv("RESEND_API_KEY") == "re-test-key"
