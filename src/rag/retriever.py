import os
from pathlib import Path

from src.config import settings
from src.rag.embedder import EmbeddingService
from src.rag.models import Chunk, RetrievalResult
from src.rag.store import ChromaStore
from src.rag.loader import load_documents_from_directory
from src.rag.chunker import RecursiveCharacterChunker
from src.utils.logger import get_logger

logger = get_logger(__name__)

METHODOLOGY_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "knowledge" / "methodology"


class Retriever:
    def __init__(self, store: ChromaStore, embedder: EmbeddingService):
        self.store = store
        self.embedder = embedder
        self._graph_retriever = None

    @property
    def graph_retriever(self):
        if self._graph_retriever is None:
            from src.rag.graph_retriever import GraphRetriever
            self._graph_retriever = GraphRetriever()
        return self._graph_retriever

    def build_graph_index(self, force: bool = False) -> bool:
        """Build the knowledge graph index from all documents in the KB."""
        if self.store.count() == 0:
            logger.warning("KB empty, skipping graph index build")
            return False
        all_data = self.store._collection.get(include=["documents"])
        documents = all_data.get("documents", [])
        return self.graph_retriever.build_index(documents, force=force)

    def ensure_methodology_loaded(self, methodology_dir: str | None = None) -> int:
        """Load methodology docs into KB if not already present. Returns chunk count added."""
        mdir = methodology_dir or str(METHODOLOGY_DIR)
        if not os.path.isdir(mdir):
            logger.info("Methodology directory not found: %s", mdir)
            return 0

        existing_types = self.store.get_source_types()
        if "methodology" in existing_types:
            logger.info("Methodology already loaded in KB")
            return 0

        logger.info("Loading methodology documents from %s...", mdir)
        try:
            docs = load_documents_from_directory(mdir)
            if not docs:
                logger.warning("No methodology documents found in %s", mdir)
                return 0
            for doc in docs:
                doc.metadata["source_type"] = "methodology"
            chunker = RecursiveCharacterChunker()
            chunks = chunker.chunk(docs)
            embeddings = self.embedder.embed_texts([c.content for c in chunks])
            self.store.add_chunks(chunks, embeddings)
            logger.info("Methodology loaded: %d docs → %d chunks", len(docs), len(chunks))
            return len(chunks)
        except Exception as e:
            logger.error("Failed to load methodology: %s", e)
            return 0

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
        graph_context = self.graph_retriever.search(query)

        if not results and not graph_context:
            return "暂无相关历史案例。请根据通用产品知识生成PRD。"

        sections: list[str] = []

        if graph_context:
            sections.append("## 知识图谱\n" + graph_context)

        if results:
            methodology_parts: list[str] = []
            case_parts: list[str] = []
            for r in results:
                source = r.chunk.metadata.get("file_name", "未知来源")
                source_type = r.chunk.metadata.get("source_type", "user_upload")
                entry = f"- **{source}**: {r.chunk.content[:300]}..."
                if source_type == "methodology":
                    methodology_parts.append(entry)
                else:
                    case_parts.append(entry)

            if methodology_parts:
                sections.append("## 方法论参考\n" + "\n".join(methodology_parts))
            if case_parts:
                sections.append("## 相似案例\n" + "\n".join(case_parts))

        return "\n\n".join(sections) if sections else "暂无相关历史案例。"
