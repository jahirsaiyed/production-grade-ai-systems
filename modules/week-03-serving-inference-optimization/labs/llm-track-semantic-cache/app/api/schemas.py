from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=500)


class Citation(BaseModel):
    source: str
    chunk_id: int
    snippet: str


class AskResponse(BaseModel):
    question: str
    answer: str
    citations: list[Citation]
    source: str
