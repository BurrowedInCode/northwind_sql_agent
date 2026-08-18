from pydantic import BaseModel, Field


class SQLGeneration(BaseModel):
    sql: str = Field(min_length=1)
    explanation: str = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)
