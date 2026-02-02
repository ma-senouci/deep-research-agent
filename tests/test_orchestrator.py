import pytest
from unittest.mock import AsyncMock
import logic_agents.orchestrator as orchestrator_mod
from logic_agents import run_pipeline
from models.schemas import (
    AgentResult, SearchPlan, SearchTerm,
    SearchSummary, ResearchReport, DeliveryResult,
)


SEARCH_PLAN = AgentResult(success=True, data=SearchPlan(searches=[
    SearchTerm(reasoning="Core topic", query="AI agent frameworks 2025"),
    SearchTerm(reasoning="Use cases", query="multi-agent orchestration patterns"),
]))

SCOUT_RESULTS = [
    AgentResult(success=True, data=SearchSummary(
        query="AI agent frameworks 2025",
        summary="OpenAI Agents SDK is the leading framework...",
        sources=["https://openai.com/agents"],
    )),
]

REPORT = AgentResult(success=True, data=ResearchReport(
    title="AI Agent Frameworks: A Comprehensive Analysis",
    overview="The landscape of AI agent frameworks is evolving rapidly.",
    body="## Introduction\n\nAI agent frameworks have matured significantly...",
    follow_up_questions=["What are agent frameworks for?", "How do they scale?"],
))

DELIVERY_OK = AgentResult(success=True, data=DeliveryResult(sent=True, message="Sent"))
DELIVERY_FAIL = AgentResult(success=False, error="Resend API Error")


def _patch_all(monkeypatch, delivery_result=None):
    monkeypatch.setattr(orchestrator_mod, "run_strategist", AsyncMock(return_value=SEARCH_PLAN))
    monkeypatch.setattr(orchestrator_mod, "run_scouts", AsyncMock(return_value=SCOUT_RESULTS))
    monkeypatch.setattr(orchestrator_mod, "run_analyst", AsyncMock(return_value=REPORT))
    monkeypatch.setattr(
        orchestrator_mod, "run_delivery",
        AsyncMock(return_value=delivery_result or DELIVERY_OK),
    )


@pytest.mark.asyncio
async def test_run_pipeline_full_happy_path(monkeypatch):
    _patch_all(monkeypatch)
    result = await run_pipeline("AI query", ["Some answers"], "test@example.com")
    assert result.success is True
    assert result.data.title == REPORT.data.title


@pytest.mark.asyncio
async def test_run_pipeline_strategist_failure(monkeypatch):
    _patch_all(monkeypatch)
    fail = AgentResult(success=False, error="Strategist error")
    monkeypatch.setattr(orchestrator_mod, "run_strategist", AsyncMock(return_value=fail))
    result = await run_pipeline("AI query", ["answers"])
    assert result.success is False
    assert "Strategist" in result.error


@pytest.mark.asyncio
async def test_run_pipeline_all_scouts_fail_propagates_analyst_error(monkeypatch):
    _patch_all(monkeypatch)
    empty_scouts = [AgentResult(success=False, error="Search error")]
    monkeypatch.setattr(orchestrator_mod, "run_scouts", AsyncMock(return_value=empty_scouts))
    analyst_fail = AgentResult(success=False, error="No search summaries provided")
    monkeypatch.setattr(orchestrator_mod, "run_analyst", AsyncMock(return_value=analyst_fail))
    result = await run_pipeline("AI query", ["answers"])
    assert result.success is False
    assert "Analyst" in result.error


@pytest.mark.asyncio
async def test_run_pipeline_analyst_failure(monkeypatch):
    _patch_all(monkeypatch)
    fail = AgentResult(success=False, error="Synthesis error")
    monkeypatch.setattr(orchestrator_mod, "run_analyst", AsyncMock(return_value=fail))
    result = await run_pipeline("AI query", ["answers"])
    assert result.success is False
    assert "Analyst" in result.error


@pytest.mark.asyncio
async def test_run_pipeline_delivery_failure_does_not_fail_pipeline(monkeypatch):
    _patch_all(monkeypatch, delivery_result=DELIVERY_FAIL)
    result = await run_pipeline("AI query", ["answers"])
    assert result.success is True
    assert result.data is not None


@pytest.mark.asyncio
async def test_run_pipeline_calls_status_callback(monkeypatch):
    _patch_all(monkeypatch)
    statuses = []
    await run_pipeline("AI query", ["answers"], status_callback=lambda s: statuses.append(s))
    assert any("Plan" in s for s in statuses)
    assert any("Search" in s for s in statuses)
    assert any("Synth" in s for s in statuses)
    assert any("Send" in s for s in statuses)


@pytest.mark.asyncio
async def test_run_pipeline_unexpected_exception(monkeypatch):
    monkeypatch.setattr(orchestrator_mod, "run_strategist", AsyncMock(side_effect=RuntimeError("boom")))
    result = await run_pipeline("AI query", ["answers"])
    assert result.success is False
    assert "boom" in result.error
