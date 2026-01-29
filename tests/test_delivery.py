import pytest
import resend
from unittest.mock import AsyncMock, Mock
from agents import Runner
from logic_agents import run_delivery
from logic_agents.delivery import _EmailContent
from models.schemas import DeliveryResult, ResearchReport


RESEARCH_REPORT = ResearchReport(
    title="AI Agent Frameworks: A Comprehensive Analysis",
    overview="The landscape of AI agent frameworks is evolving rapidly.",
    body="## Introduction\n\nAI agent frameworks have matured significantly...",
    follow_up_questions=[
        "How do agent frameworks handle state persistence?",
        "What are the benchmarks for parallel agent execution?",
    ],
)

EMAIL_HTML = "<h2>Report</h2><p>Content</p>"


def _mock_delivery(monkeypatch, html):
    mock_result = AsyncMock()
    mock_result.final_output = _EmailContent(html=html)
    monkeypatch.setattr(Runner, "run", AsyncMock(return_value=mock_result))
    monkeypatch.setattr(resend.Emails, "send", Mock(return_value={"id": "test-id"}))
    monkeypatch.setenv("RESEND_API_KEY", "re_test_key")


@pytest.mark.asyncio
async def test_run_delivery_returns_delivery_result(monkeypatch):
    _mock_delivery(monkeypatch, EMAIL_HTML)
    result = await run_delivery(RESEARCH_REPORT, recipient_email="test@example.com")
    assert result.success is True
    assert result.data.sent is True
    assert result.error is None


@pytest.mark.asyncio
async def test_run_delivery_wraps_exceptions(monkeypatch):
    monkeypatch.setenv("RESEND_API_KEY", "re_test_key")
    monkeypatch.setattr(Runner, "run", AsyncMock(side_effect=Exception("LLM Error")))
    result = await run_delivery(RESEARCH_REPORT, recipient_email="test@example.com")
    assert result.success is False
    assert result.data is None
    assert "LLM Error" in result.error


@pytest.mark.asyncio
async def test_run_delivery_handles_resend_failure(monkeypatch):
    monkeypatch.setenv("RESEND_API_KEY", "re_test_key")
    mock_result = AsyncMock()
    mock_result.final_output = _EmailContent(html=EMAIL_HTML)
    monkeypatch.setattr(Runner, "run", AsyncMock(return_value=mock_result))
    monkeypatch.setattr(resend.Emails, "send", Mock(side_effect=Exception("Resend API Error")))
    result = await run_delivery(RESEARCH_REPORT, recipient_email="test@example.com")
    assert result.success is False
    assert "Resend API Error" in result.error


@pytest.mark.asyncio
async def test_run_delivery_rejects_missing_api_key(monkeypatch):
    monkeypatch.delenv("RESEND_API_KEY", raising=False)
    result = await run_delivery(RESEARCH_REPORT, recipient_email="test@example.com")
    assert result.success is False
    assert "RESEND_API_KEY" in result.error
