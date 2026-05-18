import uuid
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import settings
from src.rag.models import Document, Chunk
from src.utils.logger import get_logger

logger = get_logger(__name__)


class RecursiveCharacterChunker:
    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        separators: list[str] | None = None,
    ):
        self.chunk_size = chunk_size or settings.default_chunk_size
        self.chunk_overlap = chunk_overlap or settings.default_chunk_overlap
        self.separators = separators or [
            "\n## ",
            "\n# ",
            "\n### ",
            "\n\n",
            "\n",
            "。",
            ".",
            " ",
        ]
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
            keep_separator=True,
        )

    def chunk(self, documents: list[Document]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for doc in documents:
            texts = self._splitter.split_text(doc.content)
            for i, text in enumerate(texts):
                chunk_id = str(uuid.uuid4())
                meta = {**doc.metadata, "chunk_index": i, "chunk_count": len(texts)}
                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        content=text,
                        metadata=meta,
                        source_doc_id=doc.doc_id,
                    )
                )
        logger.info("Split %d documents into %d chunks", len(documents), len(chunks))
        return chunks
