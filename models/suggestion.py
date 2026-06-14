"""
Models: Domain Entities
Pure dataclasses — no framework dependencies.
Previously at domain/entities/suggestion.py.
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class Suggestion:
    text: str
    tag: str
    labels: str
    concern: str
    ce_score: float = 0.0


@dataclass
class SourceCard:
    concern: str
    title: str
    tag: str
    all_labels: str
    suggestions: List[str] = field(default_factory=list)


@dataclass
class QueryResult:
    answer: str
    sources: List[SourceCard] = field(default_factory=list)
