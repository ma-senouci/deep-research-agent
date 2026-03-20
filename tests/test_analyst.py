import pytest
from unittest.mock import AsyncMock
from agents import Runner
import logic_agents.analyst as analyst_mod
from logic_agents import run_analyst
from models.schemas import ResearchReport, SearchSummary


SEARCH_SUMMARIES = [
    SearchSummary(
        query="AI agent frameworks 2025",
        summary="OpenAI Agents SDK provides primitives for building multi-agent systems...",
        sources=["https://openai.com/agents"],
    ),
    SearchSummary(
        query="multi-agent orchestration patterns",
        summary="Common patterns include agents-as-tools, parallel execution, and handoffs...",
        sources=["https://example.com/patterns"],
    ),
]

RESEARCH_REPORT = ResearchReport(
    title="AI Agent Frameworks: A Comprehensive Analysis",
    overview="The landscape of AI agent frameworks is evolving rapidly with OpenAI's Agents SDK leading the charge.",
    body="## Introduction\n\nAI agent frameworks have matured significantly...\n\n## Key Findings\n\n...",
    follow_up_questions=[
        "How do agent frameworks handle state persistence across sessions?",
        "What are the performance benchmarks for parallel agent execution?",
        "How does the OpenAI Agents SDK compare to LangGraph for complex workflows?",
    ],
)


def _mock_analyst(monkeypatch, report):
    mock_result = AsyncMock()
    mock_result.final_output = report
    monkeypatch.setattr(Runner, "run", AsyncMock(return_value=mock_result))


@pytest.mark.asyncio
async def test_run_analyst_returns_research_report(monkeypatch):
    _mock_analyst(monkeypatch, RESEARCH_REPORT)
    result = await run_analyst(SEARCH_SUMMARIES)
    assert result.success is True
    assert result.data == RESEARCH_REPORT
    assert result.error is None


@pytest.mark.asyncio
async def test_run_analyst_wraps_exceptions(monkeypatch):
    monkeypatch.setattr(Runner, "run", AsyncMock(side_effect=Exception("LLM Error")))
    result = await run_analyst(SEARCH_SUMMARIES)
    assert result.success is False
    assert result.data is None
    assert "LLM Error" in result.error


@pytest.mark.asyncio
async def test_run_analyst_rejects_empty_summaries():
    result = await run_analyst([])
    assert result.success is False
    assert "No search summaries" in result.error


@pytest.mark.asyncio
async def test_run_analyst_uses_configured_model(monkeypatch):
    mock_result = AsyncMock()
    mock_result.final_output = RESEARCH_REPORT
    runner_mock = AsyncMock(return_value=mock_result)
    monkeypatch.setattr(Runner, "run", runner_mock)
    monkeypatch.setenv("ANALYST_MODEL", "gpt-4.1-mini")

    result = await run_analyst(SEARCH_SUMMARIES)

    assert result.success is True
    agent = runner_mock.await_args.args[0]
    assert agent.model == "gpt-4.1-mini"
