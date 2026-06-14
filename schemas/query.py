"""
Schemas: API Request & Response Models
Pydantic models for request validation and response serialisation.
Previously at api/schemas/query.py.
"""
from typing import List

from pydantic import BaseModel, Field


# ── Request ───────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        examples=["I feel overwhelmed and anxious all the time"],
    )
    tag_filter: str = Field(
        default="All",
        description="Filter results by tag. Use /tags to get valid values.",
        examples=["All", "Anxiety", "Depression"],
    )
    n_tips: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of suggestions to retrieve (1–20).",
    )


# ── Response ──────────────────────────────────────────────────────────────────

class SourceCardResponse(BaseModel):
    concern: str
    title: str
    tag: str
    all_labels: str
    suggestions: List[str]

    model_config = {"from_attributes": True}


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceCardResponse]

    model_config = {"from_attributes": True}


class TagsResponse(BaseModel):
    tags: List[str]


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str
