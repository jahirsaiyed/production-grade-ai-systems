from pydantic import BaseModel, Field

from app.domain.qa import MAX_QUESTION_LENGTH


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=MAX_QUESTION_LENGTH)


class AskResponse(BaseModel):
    question: str
    answer: str
    source: str
