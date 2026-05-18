"""智能PRD自动生成平台 — Streamlit UI"""

import json
import re
import sys
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

sys.path.insert(0, str(Path(__file__).parent))

from src.config import settings
from src.rag.embedder import EmbeddingService
from src.rag.retriever import Retriever
from src.rag.store import ChromaStore
from src.output.exporter import export_to_markdown_file, generate_shareable_filename
from src.output.json_to_markdown import convert_to_prd_markdown
from src.output.validator import parse_and_validate
from src.utils.logger import setup_logging
from src.workflow.graph import run_workflow
from src.workflow.state import STAGE_ORDER

setup_logging()

st.set_page_config(
    page_title="智能PRD自动生成平台",
    page_icon="📋",
    layout="wide",
)

# --- CSS for Mermaid ---
st.markdown(
    """
<style>
.mermaid { text-align: center; }
.stExpander { border: 1px solid #e0e0e0; border-radius: 8px; margin-bottom: 8px; }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def init_rag() -> Retriever:
    embedder = EmbeddingService()
    store = ChromaStore()
    return Retriever(store, embedder)


def render_sidebar() -> None:
    with st.sidebar:
        st.header("⚙️ 配置")
        st.caption(f"LLM 模型: {settings.deepseek_model}")
        st.caption(f"嵌入模型: {settings.embedding_model_name}")

        st.divider()

        st.header("📚 知识库")
        try:
            retriever = init_rag()
            count = retriever.store.count()
            st.metric("已索引文档片段", count)
            if count == 0:
                st.warning("知识库为空，请先导入样本")
                if st.button("从 prd_samples/ 初始化知识库"):
                    seed_knowledge_base()
        except Exception as e:
            st.error(f"知识库连接失败: {e}")

        st.divider()
        st.caption("Made with LangGraph + DeepSeek")


def seed_knowledge_base() -> None:
    from src.rag.chunker import RecursiveCharacterChunker
    from src.rag.loader import load_documents_from_directory

    with st.spinner("正在导入 PRD 样本..."):
        retriever = init_rag()
        docs = load_documents_from_directory(str(settings.samples_dir))
        if not docs:
            st.warning(f"未在 {settings.samples_dir} 中找到文档")
            return
        chunker = RecursiveCharacterChunker()
        chunks = chunker.chunk(docs)
        embeddings = retriever.embedder.embed_texts([c.content for c in chunks])
        retriever.store.add_chunks(chunks, embeddings)
        st.success(f"已导入 {len(chunks)} 个文档片段")
        st.rerun()


def display_stage_result(stage_key: str, data: dict, index: int) -> None:
    stage_labels = {
        "requirement_analysis": "需求分析",
        "architecture_design": "架构设计",
        "process_flow": "流程梳理",
        "final_prd_json": "文档定稿",
    }
    label = stage_labels.get(stage_key, stage_key)
    icon = ["🔍", "🏗️", "🔄", "📝"][index] if index < 4 else "📄"

    if index < 2:
        stage_desc = f"{icon} 阶段 1（并行 {index + 1}/2）：{label} — 点击展开查看"
    else:
        stage_desc = f"{icon} 阶段 {index + 1}：{label} — 点击展开查看"

    with st.expander(stage_desc, expanded=(stage_key == "final_prd_json")):
        if stage_key == "process_flow" and data:
            _display_mermaid_diagrams(data)
        st.json(data)


def _display_mermaid_diagrams(data: dict) -> None:
    diagrams = data.get("mermaid_diagrams", [])
    for d in diagrams:
        code = d.get("mermaid_code", "")
        name = d.get("diagram_name", "流程图")
        if code:
            st.caption(f"📊 {name}")
            _render_mermaid_diagram(code)


def main() -> None:
    st.title("📋 智能PRD自动生成平台")
    st.caption("输入一句话产品想法，AI 自动生成符合大厂规范的完整 PRD 文档")

    render_sidebar()

    # --- Input ---
    col1, col2 = st.columns([3, 1])
    with col1:
        product_idea = st.text_area(
            "💡 产品想法",
            placeholder="例如：一个帮助职场新人做职业规划的AI助手，根据用户技能和目标推荐学习路径，并生成个性化周计划。",
            height=100,
            key="product_idea",
        )
    with col2:
        supplementary_info = st.text_area(
            "📎 补充说明（可选）",
            placeholder="目标用户、特殊场景、技术约束等...",
            height=100,
            key="supplementary_info",
        )

    # --- Generate ---
    gen_col, _ = st.columns([1, 3])
    with gen_col:
        generate_clicked = st.button(
            "🚀 生成 PRD",
            type="primary",
            use_container_width=True,
            disabled=not product_idea,
        )

    if generate_clicked:
        _run_pipeline(product_idea, supplementary_info)

    # --- Show results from session state ---
    if "workflow_result" in st.session_state:
        _display_results()


def _run_pipeline(product_idea: str, supplementary_info: str) -> None:
    st.divider()

    # Step 1: RAG retrieval
    with st.status("🔍 正在检索相关历史案例...", expanded=True) as status:
        retriever = init_rag()
        retrieved_context = retriever.search_as_context(product_idea)
        st.markdown(f"**检索结果**（将注入到生成流程）：\n\n{retrieved_context[:800]}..."
                    if len(retrieved_context) > 800
                    else f"**检索结果**：\n\n{retrieved_context}")
        status.update(label="✅ 检索完成", state="complete")

    # Step 2: Run workflow (3 stages, stage 1 runs requirement + architecture in parallel)
    progress_bar = st.progress(0, text="准备生成...")
    stage_status = st.empty()

    stage_names = ["需求分析 + 架构设计（并行）", "流程梳理", "文档定稿"]
    for i, stage_label in enumerate(stage_names):
        progress_bar.progress(i * 33, text=f"进行中：{stage_label}...")
        stage_status.info(f"⏳ 正在执行第 {i + 1}/3 阶段：{stage_label}")

    progress_bar.progress(80, text="执行中：文档定稿...")

    try:
        result = run_workflow(
            product_idea=product_idea,
            supplementary_info=supplementary_info,
            retrieved_context=retrieved_context,
        )

        if result.get("error_message"):
            st.error(f"生成过程中出现错误: {result['error_message']}")
            return

        final_json = result.get("final_prd_json", {})
        if not final_json:
            st.error("生成失败：未能获得最终文档")
            return

        progress_bar.progress(90, text="生成 Markdown...")

        prd_markdown = convert_to_prd_markdown(final_json)
        result["final_prd_markdown"] = prd_markdown

        progress_bar.progress(100, text="✅ 生成完成！")
        st.session_state.workflow_result = result
        st.rerun()

    except Exception as e:
        progress_bar.progress(100, text="❌ 生成失败")
        st.error(f"生成失败: {e}")


def _render_markdown_with_mermaid(text: str) -> None:
    """Render markdown text, converting ```mermaid blocks to rendered diagrams."""
    pattern = r"```mermaid\s*\n(.*?)\n```"
    parts = re.split(pattern, text, flags=re.DOTALL)

    for i, part in enumerate(parts):
        if i % 2 == 0:
            # Regular markdown text
            if part.strip():
                st.markdown(part)
        else:
            # Mermaid diagram code
            _render_mermaid_diagram(part.strip())


def _render_mermaid_diagram(code: str) -> None:
    """Render a single Mermaid diagram using the Mermaid JS library."""
    html = f"""<!DOCTYPE html>
<html>
<head>
<script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
<script>mermaid.initialize({{startOnLoad:true, theme:'default'}});</script>
<style>
body {{ margin: 0; padding: 8px; display: flex; justify-content: center; }}
</style>
</head>
<body>
<div class="mermaid">
{code}
</div>
</body>
</html>"""
    components.html(html, height=400, scrolling=False)


def _display_results() -> None:
    st.divider()
    st.header("📄 生成结果")

    result = st.session_state.workflow_result

    # Stage outputs
    stage_keys = ["requirement_analysis", "architecture_design", "process_flow"]
    for i, key in enumerate(stage_keys):
        data = result.get(key)
        if data:
            display_stage_result(key, data, i)

    # Final PRD
    final_json = result.get("final_prd_json", {})
    if final_json:
        with st.expander("📝 阶段 4：文档定稿（JSON）— 点击展开查看", expanded=False):
            st.json(final_json)

    # Final Markdown
    prd_md = result.get("final_prd_markdown", "")
    if prd_md:
        st.divider()
        st.subheader("📄 最终 PRD 文档预览")
        _render_markdown_with_mermaid(prd_md)

        # Download
        col1, col2 = st.columns(2)
        with col1:
            filename = generate_shareable_filename(
                final_json.get("version_record", {}).get("product_name", "PRD")
            )
            st.download_button(
                "⬇️ 下载 Markdown",
                data=prd_md,
                file_name=filename,
                mime="text/markdown",
                use_container_width=True,
            )
        with col2:
            try:
                from src.output.exporter import export_to_pdf
                import tempfile
                import os

                if st.button("⬇️ 导出 PDF", use_container_width=True):
                    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                        pdf_path = export_to_pdf(prd_md, tmp.name)
                        with open(pdf_path, "rb") as f:
                            st.download_button(
                                "📥 点击下载 PDF",
                                data=f,
                                file_name=filename.replace(".md", ".pdf"),
                                mime="application/pdf",
                            )
            except Exception:
                st.caption("PDF 导出需要 weasyprint 支持")


if __name__ == "__main__":
    main()
