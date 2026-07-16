"""Pydantic models for retrieval, answers, and datasets."""

from pydantic import BaseModel, Field
from typing import List, Optional
import uuid

class MinimalSource(BaseModel):
    """A source chunk location used in retrieval outputs."""

    file_path: str
    first_character_index: int
    last_character_index: int


class MinimalSearchResults(BaseModel):
    """Search results for a single question."""

    question_id: str
    question: str
    retrieved_sources: List[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """An answered question with retrieved sources."""
    answer: str


class StudentSearchResults(BaseModel):
    """Dataset of search-only outputs."""

    search_results: List[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    """Dataset of search outputs paired with generated answers."""

    search_results: List[MinimalAnswer]
    k: int


class UnansweredQuestion(BaseModel):
    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """Answered question"""
    sources: List[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """Container for all retrieval-augmented questions in a dataset."""

    rag_questions:  List[AnsweredQuestion | UnansweredQuestion]