from pydantic import BaseModel
from typing import List, Optional

class MinimalSource(BaseModel):
    file_path: str
    first_character_index: int
    last_character_index: int

class MinimalSearchResults(BaseModel):
    question_id: str
    question: str
    retrieved_sources: List[MinimalSource]

class MinimalAnswer(BaseModel):
    question_id: str
    question: str
    retrieved_sources: List[MinimalSource]
    answer: str

class StudentSearchResults(BaseModel):
    search_results: List[MinimalSearchResults]
    k: int

class StudentSearchResultsAndAnswer(BaseModel):
    search_results: List[MinimalAnswer]
    k: int

class AnsweredQuestion(BaseModel):
    question_id: str
    question: str
    sources: Optional[List[MinimalSource]] = None

class RagDataset(BaseModel):
    rag_questions: List[AnsweredQuestion]