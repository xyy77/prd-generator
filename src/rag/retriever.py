from src.config import settings
from src.rag.embedder import EmbeddingService
from src.rag.models import Chunk, RetrievalResult
from src.rag.store import ChromaStore
from src.utils.logger import get_logger

logger = get_logger(__name__)


class Retriever:
    def __init__(self, store: ChromaStore, embedder: EmbeddingService):
        self.store = store
        self.embedder = embedder

    def search(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        k = top_k or settings.max_retrieved_docs
        if self.store.count() == 0:
            logger.warning("Knowledge base is empty, returning no results")
            return []

        query_emb = self.embedder.embed_query(query)
        docs, metas, distances = self.store.query(query_emb, top_k=k)

        results: list[RetrievalResult] = []
        for i, (doc_content, meta) in enumerate(zip(docs, metas)):
            distance = distances[i] if i < len(distances) else 0.0
            score = 1.0 - distance  # cosine distance -> similarity
            chunk = Chunk(
                chunk_id=meta.get("chunk_id", ""),
                content=doc_content,
                metadata=meta,
                source_doc_id=meta.get("source_doc_id", ""),
            )
            results.append(RetrievalResult(chunk=chunk, score=score, distance=distance))
        logger.info("Retrieved %d results for query: %s...", len(results), query[:50])
        return results

    def search_as_context(self, query: str, top_k: int | None = None) -> str:
        results = self.search(query, top_k=top_k)
        if not results:
            return "暂无相关历史案例。请根据通用产品知识生成PRD。"

        parts: list[str] = []
        for i, r in enumerate(results, 1):
            source = r.chunk.metadata.get("file_name", "未知来源")
            parts.append(f"[参考案例 {i}]（来源：{source}）\n{r.chunk.content}\n")
        return "\n---\n".join(parts)
