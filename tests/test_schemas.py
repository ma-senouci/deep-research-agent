import pytest
from pydantic import ValidationError
from models.schemas import ClarificationResult


def test_clarification_validation():
    """Ensure exactly 3 questions are required for clarification."""
    assert ClarificationResult(questions=["Q1", "Q2", "Q3"])
    
    with pytest.raises(ValidationError, match="Exactly 3 clarifying questions are required"):
        ClarificationResult(questions=["Q1", "Q2"])
