from typing import Generic, TypeVar
from pydantic import BaseModel, Field, field_validator

T = TypeVar("T")

class AgentResult(BaseModel, Generic[T]):
    success: bool
    data: T | None = None
    error: str | None = None

class ClarificationResult(BaseModel):
    questions: list[str] = Field(..., description="Exactly 3 targeted questions to refine user intent")

    @field_validator("questions")
    @classmethod
    def validate_count(cls, v: list[str]) -> list[str]:
        if len(v) != 3:
            raise ValueError("Exactly 3 clarifying questions are required")
        return v

class SearchTerm(BaseModel):
    reasoning: str = Field(..., description="Rationale for choosing this specific query")
    query: str = Field(..., description="Optimized search query string")

class SearchPlan(BaseModel):
    searches: list[SearchTerm]

class SearchSummary(BaseModel):
    query: str
    summary: str
    sources: list[str]

class ResearchReport(BaseModel):
    title: str = Field(..., description="Concise, descriptive headline capturing the core finding")
    overview: str = Field(..., description="Executive overview of key insights in 2-3 sentences")
    body: str = Field(..., description="In-depth analysis in structured markdown with sections and citations")
    follow_up_questions: list[str] = Field(..., description="3 specific questions that extend or deepen this research")

class DeliveryResult(BaseModel):
    sent: bool
    message: str
