import tempfile
from pathlib import Path

import pytest

from src.rag.store import ChromaStore


@pytest.fixture
def temp_store():
    with tempfile.TemporaryDirectory() as tmp:
        store = ChromaStore(persist_dir=tmp, collection_name="test_prd")
        yield store
        store.reset()


class TestChromaStore:
    def test_initial_count_zero(self, temp_store):
        assert temp_store.count() == 0

    def test_add_and_query(self, temp_store, sample_chunks):
        embeddings = [[0.1] * 768 for _ in sample_chunks]
        temp_store.add_chunks(sample_chunks, embeddings)
        assert temp_store.count() == 2

        query_emb = [0.1] * 768
        docs, metas, distances = temp_store.query(query_emb, top_k=2)
        assert len(docs) == 2

    def test_delete_by_source(self, temp_store, sample_chunks):
        embeddings = [[0.1] * 768 for _ in sample_chunks]
        temp_store.add_chunks(sample_chunks, embeddings)
        assert temp_store.count() == 2

    def test_reset(self, temp_store, sample_chunks):
        embeddings = [[0.1] * 768 for _ in sample_chunks]
        temp_store.add_chunks(sample_chunks, embeddings)
        temp_store.reset()
        assert temp_store.count() == 0
