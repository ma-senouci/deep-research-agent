import asyncio
import pytest
from unittest.mock import AsyncMock
from agents import Runner
from logic_agents import scout_agent, run_scout, run_scouts
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


@pytest.mark.asyncio
async def test_run_scouts_collects_partial_results(monkeypatch):
    async def mock_run(agent, query):
        if query == "q2":
            raise Exception("Search failed")
        mock_result = AsyncMock()
        mock_result.final_output = SEARCH_SUMMARY
        return mock_result

    monkeypatch.setattr(Runner, "run", mock_run)

    results = await run_scouts(["q1", "q2", "q3"])

    assert len(results) == 3
    assert results[0].success is True
    assert results[1].success is False
    assert results[2].success is True


@pytest.mark.asyncio
async def test_run_scouts_reports_progress(monkeypatch):
    _mock_scout(monkeypatch, SEARCH_SUMMARY)
    reports = []

    results = await run_scouts(["q1", "q2", "q3"], status_callback=reports.append)

    assert len(reports) == 3
    assert reports[0] == "1/3 searches complete"
    assert reports[1] == "2/3 searches complete"
    assert reports[2] == "3/3 searches complete"


@pytest.mark.asyncio
async def test_run_scouts_no_callback(monkeypatch):
    _mock_scout(monkeypatch, SEARCH_SUMMARY)
    results = await run_scouts(["q1", "q2"])
    assert all(r.success is True for r in results)


@pytest.mark.asyncio
async def test_run_scouts_all_success(monkeypatch):
    _mock_scout(monkeypatch, SEARCH_SUMMARY)
    results = await run_scouts(["q1", "q2", "q3"])
    assert len(results) == 3
    assert all(r.success is True for r in results)


@pytest.mark.asyncio
async def test_run_scouts_empty_queries(monkeypatch):
    _mock_scout(monkeypatch, SEARCH_SUMMARY)
    results = await run_scouts([])
    assert results == []


@pytest.mark.asyncio
async def test_run_scouts_preserves_result_order(monkeypatch):
    async def mock_run(agent, query):
        mock_result = AsyncMock()
        mock_result.final_output = SearchSummary(
            query=query, summary=f"Summary for {query}", sources=[]
        )
        return mock_result

    monkeypatch.setattr(Runner, "run", mock_run)

    queries = ["alpha", "beta", "gamma"]
    results = await run_scouts(queries)

    for i, q in enumerate(queries):
        assert results[i].data.query == q
