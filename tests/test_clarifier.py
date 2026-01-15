import pytest
from unittest.mock import AsyncMock
from agents import Runner
from logic_agents.clarifier import run_clarifier
from models.schemas import ClarificationResult


@pytest.mark.asyncio
async def test_run_clarifier_returns_three_questions(monkeypatch):
    mock_result = AsyncMock()
    mock_result.final_output = ClarificationResult(
        questions=[
            "What specific industries are you interested in?",
            "What is the desired timeframe for the research?",
            "Do you need technical or market-focused insights?",
        ]
    )
    monkeypatch.setattr(Runner, "run", AsyncMock(return_value=mock_result))

    result = await run_clarifier("I need research on AI agents.")

    assert result.success is True
    assert len(result.data.questions) == 3
    assert result.error is None


@pytest.mark.asyncio
async def test_run_clarifier_wraps_exceptions(monkeypatch):
    monkeypatch.setattr(Runner, "run", AsyncMock(side_effect=Exception("API Error")))

    result = await run_clarifier("I need research on AI agents.")

    assert result.success is False
    assert result.data is None
    assert "API Error" in result.error


@pytest.mark.asyncio
async def test_run_clarifier_rejects_empty_query():
    result = await run_clarifier("")

    assert result.success is False
    assert "empty" in result.error.lower()

