"""
Utils: Text Processing Helpers
Shared text utilities used across services.
"""
import re


def clean_template_tags(text: str) -> str:
    """Remove {{...}} template placeholders from text."""
    return re.sub(r"\{\{.*?\}\}", "", text)


def normalise_whitespace(text: str) -> str:
    """Collapse multiple whitespace characters into a single space."""
    return re.sub(r"\s+", " ", text).strip()


def truncate_to_sentences(text: str, max_sentences: int = 3) -> str:
    """Truncate *text* to the first *max_sentences* complete sentences."""
    sentences = re.split(r"(?<=[.!])\s+", text)
    return " ".join(sentences[:max_sentences])
