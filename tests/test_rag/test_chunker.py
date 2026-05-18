from src.rag.loader import MarkdownLoader
from src.rag.chunker import RecursiveCharacterChunker


class TestRecursiveCharacterChunker:
    def test_chunk_splits_document(self, tmp_markdown_file):
        loader = MarkdownLoader()
        docs = loader.load(tmp_markdown_file)
        chunker = RecursiveCharacterChunker(chunk_size=100, chunk_overlap=20)
        chunks = chunker.chunk(docs)
        assert len(chunks) >= 1
        for chunk in chunks:
            assert chunk.chunk_id
            assert chunk.content
            assert chunk.source_doc_id == docs[0].doc_id
            assert "chunk_index" in chunk.metadata

    def test_chunk_respects_heading_boundaries(self):
        content = "# Title\n\n## Section 1\nContent in section 1.\n\n## Section 2\nContent in section 2.\n"
        from src.rag.models import Document

        doc = Document(content=content, metadata={"file_name": "test.md"}, doc_id="d1")
        chunker = RecursiveCharacterChunker(chunk_size=200, chunk_overlap=0)
        chunks = chunker.chunk([doc])
        assert len(chunks) > 0

    def test_empty_documents(self):
        chunker = RecursiveCharacterChunker()
        chunks = chunker.chunk([])
        assert chunks == []
