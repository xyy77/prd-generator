import pytest
from unittest.mock import MagicMock


@pytest.fixture
def sample_markdown_content() -> str:
    return """# 产品需求文档：AI 写作助手

## 一、背景与目标
写作助手帮助用户快速生成文案。

## 二、用户画像
- 职场人士，需要写周报
- 学生，需要写论文
"""


@pytest.fixture
def sample_chunks(sample_markdown_content: str) -> list:
    from src.rag.models import Chunk

    return [
        Chunk(
            chunk_id="c1",
            content="写作助手帮助用户快速生成文案。",
            metadata={"file_name": "test.md", "chunk_index": 0},
            source_doc_id="doc1",
        ),
        Chunk(
            chunk_id="c2",
            content="职场人士，需要写周报。学生，需要写论文。",
            metadata={"file_name": "test.md", "chunk_index": 1},
            source_doc_id="doc1",
        ),
    ]


@pytest.fixture
def mock_llm_client():
    client = MagicMock()
    client.chat.return_value = '{"result": "ok"}'
    client.chat_with_json_mode.return_value = '{"result": "ok"}'
    return client


@pytest.fixture
def tmp_markdown_file(tmp_path, sample_markdown_content):
    file_path = tmp_path / "test_prd.md"
    file_path.write_text(sample_markdown_content, encoding="utf-8")
    return str(file_path)
