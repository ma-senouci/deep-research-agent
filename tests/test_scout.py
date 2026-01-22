import asyncio
import pytest
from unittest.mock import AsyncMock
from agents import Runner
from logic_agents import scout_agent, run_scout
from models.schemas import SearchSummary


SEARCH_SUMMARY = SearchSummary(
    query="AI agent frameworks 2025",
    summary="OpenAI Agents SDK provides primitives for agents-as-tools and parallel execution...",
    sources=["https://openai.com/agents", "https://example.com/multi-agent-survey"],
)


def _mock_scout(monkeypatch, summary):
    mock_result = AsyncMock()
    mock_result.final_output = summary
    monkeypatch.setattr(Runner, "run", AsyncMock(return_value=mock_result))


def test_scout_agent_forces_tool_use():
    assert scout_agent.model_settings.tool_choice == "required"


@pytest.mark.asyncio
async def test_run_scout_returns_search_summary(monkeypatch):
    _mock_scout(monkeypatch, SEARCH_SUMMARY)

    result = await run_scout("AI agent frameworks 2025")

    assert result.success is True
    assert result.data == SEARCH_SUMMARY
    assert result.error is None


@pytest.mark.asyncio
async def test_run_scout_wraps_exceptions(monkeypatch):
    monkeypatch.setattr(Runner, "run", AsyncMock(side_effect=Exception("API Error")))

    result = await run_scout("AI agent frameworks 2025")

    assert result.success is False
    assert result.data is None
    assert "API Error" in result.error


@pytest.mark.asyncio
async def test_run_scout_concurrent_execution(monkeypatch):
    _mock_scout(monkeypatch, SEARCH_SUMMARY)
    queries = ["AI agent frameworks 2025", "multi-agent systems", "LangChain alternatives"]

    results = await asyncio.gather(*[run_scout(q) for q in queries])

    assert all(r.success is True for r in results)
