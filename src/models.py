"""Pydantic models for retrieval, answers, and datasets."""

from pydantic import BaseModel
from typing import List, Optional


class MinimalSource(BaseModel):
    """A source chunk location used in retrieval outputs."""

    file_path: str
    first_character_index: int
    last_character_index: int


class MinimalSearchResults(BaseModel):
    """Search results for a single question."""

    question_id: str
    question_str: str
    retrieved_sources: List[MinimalSource]


class MinimalAnswer(BaseModel):
    """An answered question with retrieved sources."""

    question_id: str
    question_str: str
    retrieved_sources: List[MinimalSource]
    answer: str


class StudentSearchResults(BaseModel):
    """Dataset of search-only outputs."""

    search_results: List[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    """Dataset of search outputs paired with generated answers."""

    search_results: List[MinimalAnswer]
    k: int


class AnsweredQuestion(BaseModel):
    """Ground-truth question with optional source spans."""

    question_id: str
    question: str
    sources: Optional[List[MinimalSource]] = None


class RagDataset(BaseModel):
    """Container for all retrieval-augmented questions in a dataset."""

    rag_questions: List[AnsweredQuestion]
