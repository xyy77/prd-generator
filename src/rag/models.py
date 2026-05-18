from pydantic import BaseModel, Field


class Document(BaseModel):
    content: str
    metadata: dict = Field(default_factory=dict)
    doc_id: str = ""


class Chunk(BaseModel):
    chunk_id: str
    content: str
    metadata: dict = Field(default_factory=dict)
    source_doc_id: str = ""


class RetrievalResult(BaseModel):
    chunk: Chunk
    score: float
    distance: float
