import pytest

from src.rag.embedder import EmbeddingService


class TestEmbeddingService:
    @pytest.fixture(scope="class")
    def embedder(self):
        return EmbeddingService()

    def test_embed_query_returns_list(self, embedder):
        vec = embedder.embed_query("AI助手产品")
        assert isinstance(vec, list)
        assert len(vec) == embedder.dimension
        assert all(isinstance(v, float) for v in vec)

    def test_embed_texts_returns_list_of_lists(self, embedder):
        texts = ["产品需求文档", "AI写作助手"]
        vecs = embedder.embed_texts(texts)
        assert len(vecs) == 2
        assert len(vecs[0]) == embedder.dimension

    def test_embed_texts_empty(self, embedder):
        vecs = embedder.embed_texts([])
        assert vecs == []

    def test_dimension_is_768_for_bge(self, embedder):
        assert embedder.dimension == 768
