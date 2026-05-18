import uuid
from pathlib import Path

from src.rag.models import Document
from src.utils.logger import get_logger

logger = get_logger(__name__)


class MarkdownLoader:
    def load(self, file_path: str) -> list[Document]:
        path = Path(file_path)
        content = path.read_text(encoding="utf-8")
        doc_id = str(uuid.uuid4())
        logger.info("Loaded markdown: %s (%d chars)", path.name, len(content))
        return [
            Document(
                content=content,
                metadata={
                    "source": str(path.resolve()),
                    "file_name": path.name,
                    "file_type": "markdown",
                    "title": path.stem,
                },
                doc_id=doc_id,
            )
        ]


class PDFLoader:
    def load(self, file_path: str) -> list[Document]:
        import fitz  # PyMuPDF

        path = Path(file_path)
        docs: list[Document] = []
        doc = fitz.open(str(path))
        for page_num, page in enumerate(doc):
            text = page.get_text()
            if not text.strip():
                continue
            doc_id = str(uuid.uuid4())
            docs.append(
                Document(
                    content=text,
                    metadata={
                        "source": str(path.resolve()),
                        "file_name": path.name,
                        "file_type": "pdf",
                        "page_num": page_num + 1,
                        "title": path.stem,
                    },
                    doc_id=doc_id,
                )
            )
        doc.close()
        logger.info("Loaded PDF: %s (%d pages)", path.name, len(docs))
        return docs


class DOCXLoader:
    def load(self, file_path: str) -> list[Document]:
        from docx import Document as DocxDocument

        path = Path(file_path)
        docx = DocxDocument(str(path))
        paragraphs: list[str] = []
        for para in docx.paragraphs:
            if para.text.strip():
                paragraphs.append(para.text)
        content = "\n\n".join(paragraphs)
        doc_id = str(uuid.uuid4())
        logger.info("Loaded DOCX: %s (%d paragraphs)", path.name, len(paragraphs))
        return [
            Document(
                content=content,
                metadata={
                    "source": str(path.resolve()),
                    "file_name": path.name,
                    "file_type": "docx",
                    "title": path.stem,
                },
                doc_id=doc_id,
            )
        ]


LOADER_MAP = {
    ".md": MarkdownLoader,
    ".markdown": MarkdownLoader,
    ".pdf": PDFLoader,
    ".docx": DOCXLoader,
}


def load_document(file_path: str) -> list[Document]:
    path = Path(file_path)
    ext = path.suffix.lower()
    loader_cls = LOADER_MAP.get(ext)
    if loader_cls is None:
        raise ValueError(f"Unsupported file type: {ext}")
    return loader_cls().load(str(path))


def load_documents_from_directory(directory: str) -> list[Document]:
    dir_path = Path(directory)
    all_docs: list[Document] = []
    for ext, loader_cls in LOADER_MAP.items():
        for file_path in dir_path.glob(f"*{ext}"):
            try:
                docs = loader_cls().load(str(file_path))
                all_docs.extend(docs)
            except Exception as e:
                logger.warning("Failed to load %s: %s", file_path.name, e)
    logger.info("Loaded %d documents from %s", len(all_docs), directory)
    return all_docs
