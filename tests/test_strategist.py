import pytest
from unittest.mock import AsyncMock
from agents import Runner
from logic_agents import run_strategist
from models.schemas import SearchPlan, SearchTerm


SEARCH_TERMS = [
    SearchTerm(reasoning="Primary angle on the research topic", query="AI agent frameworks 2025"),
    SearchTerm(reasoning="Comparative landscape", query="comparison LangChain CrewAI OpenAI agents"),
    SearchTerm(reasoning="Practical applications", query="multi-agent systems production use cases"),
]


def _mock_search_plan(monkeypatch, terms):
    mock_result = AsyncMock()
    mock_result.final_output = SearchPlan(searches=terms)
    monkeypatch.setattr(Runner, "run", AsyncMock(return_value=mock_result))


@pytest.mark.asyncio
async def test_run_strategist_returns_search_plan(monkeypatch):
    _mock_search_plan(monkeypatch, SEARCH_TERMS)

    result = await run_strategist("AI agent frameworks", ["focus on open source", "2025 only"])

    assert result.success is True
    assert len(result.data.searches) == len(SEARCH_TERMS)
    assert result.error is None


@pytest.mark.asyncio
async def test_search_terms_have_reasoning_and_query(monkeypatch):
    _mock_search_plan(monkeypatch, SEARCH_TERMS)

    result = await run_strategist("AI agent frameworks", ["focus on open source"])

    for term in result.data.searches:
        assert term.reasoning
        assert term.query


@pytest.mark.asyncio
async def test_run_strategist_wraps_exceptions(monkeypatch):
    monkeypatch.setattr(Runner, "run", AsyncMock(side_effect=Exception("API Error")))

    result = await run_strategist("AI agent frameworks", ["any scope"])

    assert result.success is False
    assert result.data is None
    assert "API Error" in result.error


@pytest.mark.asyncio
async def test_run_strategist_rejects_empty_query():
    result = await run_strategist("", ["some answers"])

    assert result.success is False
    assert "empty" in result.error.lower()
