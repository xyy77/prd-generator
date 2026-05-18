import chromadb
from chromadb.config import Settings as ChromaSettings

from src.config import settings
from src.rag.models import Chunk
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ChromaStore:
    def __init__(
        self,
        persist_dir: str | None = None,
        collection_name: str | None = None,
    ):
        self.persist_dir = str(persist_dir or settings.chroma_persist_path)
        self.collection_name = collection_name or settings.chroma_collection_name
        self._client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "ChromaDB ready: %s/%s (%d docs)",
            self.persist_dir,
            self.collection_name,
            self._collection.count(),
        )

    def add_chunks(
        self,
        chunks: list[Chunk],
        embeddings: list[list[float]],
    ) -> None:
        if not chunks:
            return
        ids = [c.chunk_id for c in chunks]
        documents = [c.content for c in chunks]
        metadatas = [c.metadata for c in chunks]
        self._collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        logger.info("Added %d chunks to collection", len(chunks))

    def query(
        self,
        query_embedding: list[float],
        top_k: int = 3,
    ) -> tuple[list[str], list[dict], list[float]]:
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        return docs, metas, distances

    def count(self) -> int:
        return self._collection.count()

    def delete_by_source(self, source_doc_id: str) -> None:
        self._collection.delete(where={"source_doc_id": source_doc_id})

    def reset(self) -> None:
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("Collection reset")
