import tempfile

import pytest

from src.rag.embedder import EmbeddingService
from src.rag.retriever import Retriever
from src.rag.store import ChromaStore


@pytest.fixture
def retriever():
    embedder = EmbeddingService()
    store = ChromaStore(persist_dir=tempfile.mkdtemp(), collection_name="test_retrieve")
    r = Retriever(store, embedder)
    return r


class TestRetriever:
    def test_search_empty_store(self, retriever):
        results = retriever.search("AI助手")
        assert results == []

    def test_search_and_format_context(self, retriever, sample_chunks):
        embeddings = retriever.embedder.embed_texts([c.content for c in sample_chunks])
        retriever.store.add_chunks(sample_chunks, embeddings)

        results = retriever.search("写作助手", top_k=2)
        assert len(results) == 2
        assert results[0].score > 0

        context = retriever.search_as_context("写作助手")
        # Context may contain graph results or retrieved documents
        assert len(context) > 0
        assert any(kw in context for kw in ["参考案例", "知识图谱", "历史案例"])
