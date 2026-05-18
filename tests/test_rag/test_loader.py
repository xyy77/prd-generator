from pathlib import Path

from src.rag.loader import MarkdownLoader, load_document, load_documents_from_directory


class TestMarkdownLoader:
    def test_load_single_file(self, tmp_markdown_file):
        loader = MarkdownLoader()
        docs = loader.load(tmp_markdown_file)
        assert len(docs) == 1
        assert "写作助手" in docs[0].content
        assert docs[0].metadata["file_name"] == "test_prd.md"
        assert docs[0].metadata["file_type"] == "markdown"

    def test_doc_id_is_unique(self, tmp_markdown_file):
        loader = MarkdownLoader()
        docs1 = loader.load(tmp_markdown_file)
        docs2 = loader.load(tmp_markdown_file)
        assert docs1[0].doc_id != docs2[0].doc_id


class TestLoadDocument:
    def test_load_by_extension(self, tmp_markdown_file):
        docs = load_document(tmp_markdown_file)
        assert len(docs) == 1

    def test_unsupported_extension_raises(self):
        try:
            load_document("test.xyz")
        except ValueError:
            pass


class TestLoadDocumentsFromDirectory:
    def test_load_all_markdown(self, tmp_path):
        for i in range(3):
            (tmp_path / f"prd_{i}.md").write_text(f"# PRD {i}\n\nContent {i}", encoding="utf-8")
        docs = load_documents_from_directory(str(tmp_path))
        assert len(docs) == 3
