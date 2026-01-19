import pytest
from unittest.mock import AsyncMock
from agents import Runner
from logic_agents import run_clarifier
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


def _mock_clarification(monkeypatch, questions):
    mock_result = AsyncMock()
    mock_result.final_output = ClarificationResult(questions=questions)
    monkeypatch.setattr(Runner, "run", AsyncMock(return_value=mock_result))


SCOPE_QUESTIONS = [
    "Which subfield of AI interests you — NLP, computer vision, or reinforcement learning?",
    "Are you looking for academic research, industry applications, or both?",
    "What timeframe should the research cover — recent breakthroughs or historical evolution?",
]


@pytest.mark.asyncio
async def test_vague_query_returns_successful_result(monkeypatch):
    _mock_clarification(monkeypatch, SCOPE_QUESTIONS)

    result = await run_clarifier("tell me about AI")

    assert result.success is True
    assert len(result.data.questions) == 3


@pytest.mark.asyncio
async def test_short_query_handled_gracefully(monkeypatch):
    _mock_clarification(monkeypatch, SCOPE_QUESTIONS)

    result = await run_clarifier("AI")

    assert result.success is True
    assert len(result.data.questions) == 3


@pytest.mark.asyncio
async def test_vague_query_questions_are_scope_narrowing(monkeypatch):
    _mock_clarification(monkeypatch, SCOPE_QUESTIONS)

    result = await run_clarifier("stuff about tech")

    for q in result.data.questions:
        assert isinstance(q, str) and len(q) > 0 and "?" in q


@pytest.mark.asyncio
async def test_run_clarifier_never_raises(monkeypatch):
    _mock_clarification(monkeypatch, SCOPE_QUESTIONS)

    result = await run_clarifier("give me info")

    assert result.success is True
    assert result.error is None
