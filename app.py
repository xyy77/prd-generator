"""智能PRD自动生成平台 — Streamlit UI"""

import json
import os
import re
import sys
import tempfile
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

sys.path.insert(0, str(Path(__file__).parent))

from src.config import settings
from src.rag.embedder import EmbeddingService
from src.rag.retriever import Retriever
from src.rag.store import ChromaStore
from src.rag.loader import load_document
from src.rag.chunker import RecursiveCharacterChunker
from src.output.exporter import export_to_markdown_file, generate_shareable_filename
from src.output.json_to_markdown import convert_to_prd_markdown
from src.utils.logger import setup_logging
from src.workflow.graph import run_workflow, run_revision, run_single_stage
from src.workflow.multi_agent.graph import run_multi_agent_workflow, run_multi_agent_revision
from src.workflow.state import STAGE_ORDER

setup_logging()

st.set_page_config(
    page_title="智能PRD自动生成平台",
    page_icon="📋",
    layout="wide",
)

st.markdown(
    """
<style>
.mermaid { text-align: center; }
.stExpander { border: 1px solid #e0e0e0; border-radius: 8px; margin-bottom: 8px; }
</style>
""",
    unsafe_allow_html=True,
)

# --- Session state init ---
if "workflow_result" not in st.session_state:
    st.session_state.workflow_result = None
if "selected_model" not in st.session_state:
    st.session_state.selected_model = settings.deepseek_model
if "temperature" not in st.session_state:
    st.session_state.temperature = settings.llm_temperature
if "generation_history" not in st.session_state:
    st.session_state.generation_history = []
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []
if "use_multi_agent" not in st.session_state:
    st.session_state.use_multi_agent = False
if "image_paths" not in st.session_state:
    st.session_state.image_paths = []
if "reflection_max_rounds" not in st.session_state:
    st.session_state.reflection_max_rounds = settings.reflection_max_rounds
if "reviewer_score_threshold" not in st.session_state:
    st.session_state.reviewer_score_threshold = settings.reviewer_score_threshold


@st.cache_resource
def init_rag() -> Retriever:
    embedder = EmbeddingService()
    store = ChromaStore()
    retriever = Retriever(store, embedder)
    retriever.ensure_methodology_loaded()
    if retriever.store.count() > 0:
        retriever.build_graph_index()
    return retriever


def _count_unique_sources() -> int:
    """Count unique source files in the knowledge base."""
    try:
        retriever = init_rag()
        all_meta = retriever.store._collection.get(include=["metadatas"])
        metadatas = all_meta.get("metadatas", [])
        if not metadatas:
            return 0
        sources = set()
        for m in metadatas:
            src = m.get("source") if m else None
            if src:
                sources.add(src)
        return len(sources)
    except Exception:
        return 0


def _get_kb_sources() -> list[str]:
    """Get sorted list of unique source file names in the knowledge base."""
    try:
        retriever = init_rag()
        all_meta = retriever.store._collection.get(include=["metadatas"])
        metadatas = all_meta.get("metadatas", [])
        if not metadatas:
            return []
        sources = sorted(set(
            m.get("source", "unknown") for m in metadatas if m
        ))
        return sources
    except Exception:
        return []


def render_sidebar() -> None:
    with st.sidebar:
        # ── Compact sidebar CSS ──
        st.markdown(
            """
<style>
[data-testid="stSidebar"] h2 { font-size: 0.92rem; padding-top: 0.3rem; margin-bottom: 0.2rem; letter-spacing: 0.02em; }
[data-testid="stSidebar"] hr { margin: 0.3rem 0; opacity: 0.3; }
[data-testid="stSidebar"] .stSelectbox label, [data-testid="stSidebar"] .stSlider label { font-size: 0.82rem; margin-bottom: 0; }
[data-testid="stSidebar"] .stSelectbox { margin-bottom: 0.3rem; }
[data-testid="stSidebar"] .stSlider { padding-top: 0; margin-bottom: 0.3rem; }
[data-testid="stSidebar"] .stButton button { padding: 0.25rem 0.5rem; font-size: 0.78rem; border-radius: 6px; }
[data-testid="stSidebar"] .stMetric label { font-size: 0.7rem; }
[data-testid="stSidebar"] .stMetric [data-testid="stMetricValue"] { font-size: 1.1rem; }
[data-testid="stSidebar"] [data-testid="stExpander"] { border: 1px solid rgba(128,128,128,0.15); border-radius: 6px; margin-bottom: 0.3rem; }
[data-testid="stSidebar"] .stCaption { font-size: 0.75rem; color: #999; }
[data-testid="stSidebar"] .stFileUploader { margin-bottom: 0.3rem; }
[data-testid="stSidebar"] .stToggle { margin-bottom: 0.3rem; }
</style>
""",
            unsafe_allow_html=True,
        )

        # ── Section 1: Generation settings ──
        st.header("⚙️ 生成设置")

        col1, col2 = st.columns(2)
        with col1:
            available = settings.available_models
            current_idx = available.index(st.session_state.selected_model) if st.session_state.selected_model in available else 0
            selected = st.selectbox(
                "模型",
                available,
                index=current_idx,
                key="sidebar_model_select",
            )
            st.session_state.selected_model = selected
        with col2:
            temp = st.slider(
                "温度",
                min_value=settings.llm_temperature_min,
                max_value=settings.llm_temperature_max,
                value=st.session_state.temperature,
                step=0.05,
                help="0=确定性，1=高创造性。PRD 推荐 0.3-0.6",
                key="sidebar_temperature_slider",
            )
            st.session_state.temperature = temp

        use_ma = st.toggle(
            "🤖 多智能体协作模式",
            value=st.session_state.use_multi_agent,
            help="4个专业Agent + 评审官协作，支持反思循环。关闭则使用经典3阶段流水线。",
            key="sidebar_multi_agent_toggle",
        )
        st.session_state.use_multi_agent = use_ma

        if use_ma:
            r1, r2 = st.columns(2)
            with r1:
                st.session_state.reflection_max_rounds = st.slider(
                    "反思轮次",
                    min_value=1, max_value=3,
                    value=st.session_state.reflection_max_rounds,
                    step=1,
                    help="评分不达标时回退修正的最大轮次",
                    key="sidebar_reflection_rounds",
                )
            with r2:
                st.session_state.reviewer_score_threshold = st.slider(
                    "通过阈值",
                    min_value=60, max_value=95,
                    value=st.session_state.reviewer_score_threshold,
                    step=5,
                    help="评分低于此值触发反思，大厂标准 80",
                    key="sidebar_reviewer_threshold",
                )

        st.divider()

        # ── Section 2: Reference materials ──
        st.header("📤 参考资料")
        st.caption("请上传参考图片（手绘图/竞品截图/用户旅程）或示例 PRD 文档（MD/PDF/DOCX），系统将自动识别类型并处理。")
        all_uploads = st.file_uploader(
            "点击上传图片或文档",
            type=["png", "jpg", "jpeg", "webp", "md", "pdf", "docx"],
            accept_multiple_files=True,
            help="图片用于多模态视觉分析（需多智能体模式）；文档自动分块索引到知识库，PDF内嵌图片也会被提取。",
            key="sidebar_all_uploads",
        )

        if all_uploads:
            images = [f for f in all_uploads if f.type.startswith("image/")]
            docs = [f for f in all_uploads if not f.type.startswith("image/")]
            if images and st.session_state.use_multi_agent:
                _process_image_uploads(images)
            elif images and not st.session_state.use_multi_agent:
                st.caption("⚠️ 图片分析需开启多智能体模式")
            if docs:
                _process_uploads(docs)

        if st.session_state.image_paths:
            for p in st.session_state.image_paths:
                st.caption(f"🖼️ {Path(p).name}")

        st.divider()

        # ── Section 3: Knowledge base (collapsible) ──
        kb_empty = False
        try:
            retriever = init_rag()
            chunk_count = retriever.store.count()
            source_count = _count_unique_sources()
            kb_empty = chunk_count == 0
        except Exception:
            chunk_count, source_count, kb_empty = 0, 0, True

        with st.expander(
            f"📚 知识库 · {chunk_count} 片段",
            expanded=kb_empty,
        ):
            if kb_empty:
                st.warning("知识库为空，请导入样本或上传文件")
            else:
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("文档片段", chunk_count)
                with col_b:
                    st.metric("源文件", source_count)

            kb_col1, kb_col2, kb_col3 = st.columns(3)
            with kb_col1:
                if st.button("🔄 重新加载", use_container_width=True, key="kb_reload"):
                    _reload_knowledge_base()
            with kb_col2:
                if st.button("🗑️ 清空", use_container_width=True, key="kb_clear"):
                    _clear_knowledge_base()
            with kb_col3:
                if st.button("🔨 重构", use_container_width=True, key="kb_rebuild"):
                    _rebuild_knowledge_base()

            sources = _get_kb_sources()
            if sources:
                for src in sources:
                    scol1, scol2 = st.columns([4, 1])
                    with scol1:
                        st.caption(f"📄 {src}")
                    with scol2:
                        if st.button("✕", key=f"del_{src}"):
                            try:
                                retriever = init_rag()
                                retriever.store.delete_by_source(src)
                            except Exception:
                                pass
                            st.rerun()

        # ── Section 4: History (collapsible) ──
        history = st.session_state.generation_history
        with st.expander(
            f"📜 历史记录 · {len(history)}",
            expanded=bool(history),
        ):
            if history:
                for i, entry in enumerate(reversed(history)):
                    label = entry.get("product_name", f"第{i+1}次生成")
                    if st.button(f"📋 {label}", use_container_width=True, key=f"hist_{i}"):
                        st.session_state.workflow_result = entry["result"]
                        st.rerun()
            else:
                st.caption("暂无历史记录")

        st.caption("Made with LangGraph + DeepSeek")


def _process_uploads(uploaded_files: list) -> None:
    """Save uploaded files, add to knowledge base, and extract embedded images."""
    from src.rag.loader import extract_document_images

    retriever = init_rag()
    chunker = RecursiveCharacterChunker()
    uploads_dir = settings.uploads_dir
    uploads_dir.mkdir(parents=True, exist_ok=True)
    images_dir = settings.uploads_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    total_added = 0
    total_images = 0
    for f in uploaded_files:
        if f.name in st.session_state.uploaded_files:
            continue

        # Save to disk
        save_path = uploads_dir / f.name
        with open(save_path, "wb") as fh:
            fh.write(f.getbuffer())
        st.session_state.uploaded_files.append(f.name)

        # Load and chunk text
        try:
            docs = load_document(str(save_path))
            if docs:
                chunks = chunker.chunk(docs)
                if chunks:
                    embeddings = retriever.embedder.embed_texts([c.content for c in chunks])
                    retriever.store.add_chunks(chunks, embeddings)
                    total_added += len(chunks)
        except Exception as e:
            st.error(f"处理 {f.name} 失败: {e}")
            continue

        # Extract embedded images from PDF
        if f.type == "application/pdf":
            try:
                extracted = extract_document_images(str(save_path), str(images_dir))
                for img_path in extracted:
                    if len(st.session_state.image_paths) >= 3:
                        break
                    if img_path not in st.session_state.image_paths:
                        st.session_state.image_paths.append(img_path)
                        total_images += 1
            except Exception as e:
                logger.warning("Failed to extract images from %s: %s", f.name, e)

    msgs = []
    if total_added > 0:
        msgs.append(f"已添加 {total_added} 个文档片段")
    if total_images > 0:
        msgs.append(f"从 PDF 提取 {total_images} 张图片")
    if msgs:
        st.toast("✅ " + "；".join(msgs))
    if total_added > 0:
        st.rerun()


def _process_image_uploads(uploaded_images: list) -> None:
    """Save uploaded images and store paths in session state."""
    uploads_dir = settings.uploads_dir / "images"
    uploads_dir.mkdir(parents=True, exist_ok=True)

    for img in uploaded_images:
        if len(st.session_state.image_paths) >= 3:
            st.warning("最多上传 3 张图片")
            break
        save_path = uploads_dir / img.name
        save_path_str = str(save_path)
        if save_path_str in st.session_state.image_paths:
            continue
        with open(save_path, "wb") as fh:
            fh.write(img.getbuffer())
        st.session_state.image_paths.append(save_path_str)

    if uploaded_images:
        st.toast(f"🖼️ 已上传 {len(uploaded_images)} 张参考图片")


def _reload_knowledge_base() -> None:
    """Clear and reload from prd_samples + uploads."""
    from src.rag.loader import load_documents_from_directory

    retriever = init_rag()
    retriever.store.reset()
    chunker = RecursiveCharacterChunker()
    total = 0

    # Load from prd_samples
    samples_dir = settings.samples_dir
    if samples_dir.exists():
        docs = load_documents_from_directory(str(samples_dir))
        if docs:
            chunks = chunker.chunk(docs)
            if chunks:
                embeddings = retriever.embedder.embed_texts([c.content for c in chunks])
                retriever.store.add_chunks(chunks, embeddings)
                total += len(chunks)

    # Load from uploads
    uploads_dir = settings.uploads_dir
    if uploads_dir.exists():
        for f in uploads_dir.iterdir():
            if f.suffix.lower() in (".md", ".pdf", ".docx"):
                try:
                    docs = load_document(str(f))
                    if docs:
                        chunks = chunker.chunk(docs)
                        if chunks:
                            embeddings = retriever.embedder.embed_texts([c.content for c in chunks])
                            retriever.store.add_chunks(chunks, embeddings)
                            total += len(chunks)
                except Exception:
                    pass

    # Re-load methodology
    method_chunks = retriever.ensure_methodology_loaded()
    if method_chunks:
        total += method_chunks

    # Rebuild graph index
    if total > 0:
        retriever.build_graph_index(force=True)

    st.toast(f"🔄 知识库已重新加载，共 {total} 个片段")
    st.rerun()


def _clear_knowledge_base() -> None:
    retriever = init_rag()
    retriever.store.reset()
    st.toast("🗑️ 知识库已清空")
    st.rerun()


def _rebuild_knowledge_base() -> None:
    """Full rebuild: clear, re-chunk from scratch."""
    _reload_knowledge_base()


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

        # Per-stage retry button
        if st.button(f"🔄 重新生成此阶段", key=f"retry_{stage_key}"):
            with st.spinner(f"正在重新生成：{label}..."):
                result = st.session_state.workflow_result
                if result:
                    new_result = run_single_stage(stage_key, result)
                    st.session_state.workflow_result = new_result
                    st.rerun()


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

    # --- Mode indicator ---
    if st.session_state.use_multi_agent:
        st.info(
            "🤖 **多智能体协作模式** — Planner 识别产品类型 → Supervisor 智能调度 → "
            "4 个专家 Agent（需求分析师、功能规划师、UX 设计师、技术顾问）协作生成 → Reviewer 评审把关"
        )
    else:
        st.info("📋 **经典流水线模式** — 需求分析 + 架构设计（并行）→ 流程梳理 → 文档定稿，3 阶段依次执行")
    if st.session_state.image_paths:
        img_names = [Path(p).name for p in st.session_state.image_paths]
        st.caption(f"🖼️ 已上传参考图片：{', '.join(img_names)}")

    # --- Input ---
    col1, col2 = st.columns([3, 1])
    with col1:
        product_idea = st.text_area(
            "💡 产品想法",
            placeholder="例如：一个帮助职场新人做职业规划的AI助手，根据用户技能和目标推荐学习路径，并生成个性化周计划。",
            height=100,
            key="input_product_idea",
        )
    with col2:
        supplementary_info = st.text_area(
            "📎 补充说明（可选）",
            placeholder="目标用户、特殊场景、技术约束等...",
            height=100,
            key="input_supplementary_info",
        )

    # --- Generate ---
    gen_col, cap_col = st.columns([1, 3])
    with gen_col:
        btn_label = "🚀 生成 PRD (多智能体)" if st.session_state.use_multi_agent else "🚀 生成 PRD"
        generate_clicked = st.button(
            btn_label,
            type="primary",
            use_container_width=True,
            disabled=not product_idea,
            key="btn_generate",
        )
    with cap_col:
        if st.session_state.use_multi_agent:
            st.caption("多 Agent 模式预计耗时 5-15 分钟，生成过程中可查看下方 CoT 面板了解实时进度。")
        else:
            st.caption("经典模式预计耗时 2-5 分钟。")

    if generate_clicked:
        if st.session_state.use_multi_agent:
            _run_multi_agent_pipeline(product_idea, supplementary_info)
        else:
            _run_pipeline(product_idea, supplementary_info)

    # --- Show results from session state ---
    if st.session_state.workflow_result:
        if st.session_state.use_multi_agent:
            _display_multi_agent_results()
        else:
            _display_results()


def _run_pipeline(product_idea: str, supplementary_info: str) -> None:
    st.divider()

    with st.status("🔍 正在检索相关历史案例...", expanded=True) as status:
        retriever = init_rag()
        retrieved_context = retriever.search_as_context(product_idea)
        preview = retrieved_context[:800] + "..." if len(retrieved_context) > 800 else retrieved_context
        st.markdown(f"**检索结果**（将注入到生成流程）：\n\n{preview}")
        status.update(label="✅ 检索完成", state="complete")

    # Show progress during generation
    progress_bar = st.progress(0, text="准备生成...")
    stage_status = st.empty()

    stage_names = ["需求分析 + 架构设计（并行）", "流程梳理", "文档定稿"]
    for i, stage_label in enumerate(stage_names):
        progress_bar.progress(i * 33, text=f"进行中：{stage_label}...")
        stage_status.info(f"⏳ 正在执行第 {i + 1}/3 阶段：{stage_label}")

    progress_bar.progress(80, text="执行中：文档定稿...")

    try:
        with st.spinner("正在生成 PRD..."):
            result = run_workflow(
                product_idea=product_idea,
                supplementary_info=supplementary_info,
                retrieved_context=retrieved_context,
                selected_model=st.session_state.selected_model,
                temperature=st.session_state.temperature,
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

        # Save to generation history
        product_name = final_json.get("version_record", {}).get("product_name", product_idea[:30])
        st.session_state.generation_history.append({
            "product_name": product_name,
            "product_idea": product_idea,
            "timestamp": __import__("datetime").datetime.now().isoformat(),
            "result": result,
        })
        # Trim history to max
        max_hist = settings.max_history
        if len(st.session_state.generation_history) > max_hist:
            st.session_state.generation_history = st.session_state.generation_history[-max_hist:]

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
            if part.strip():
                st.markdown(part)
        else:
            _render_mermaid_diagram(part.strip())


def _render_mermaid_diagram(code: str) -> None:
    """Render a Mermaid diagram with dynamic height and click-to-enlarge."""
    line_count = code.count("\n") + 1
    height = max(300, line_count * 20 + 40)
    safe_code = code.replace("\\", "\\\\").replace("`", "\\`").replace("$", "\\$")
    html = f"""<!DOCTYPE html>
<html>
<head>
<script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
<script>mermaid.initialize({{startOnLoad:true, theme:'default'}});</script>
<style>
body {{ margin: 0; padding: 8px; display: flex; justify-content: center; background: #fff; }}
.mermaid {{ cursor: pointer; }}
#mermaid-modal {{
  display: none; position: fixed; top: 0; left: 0; width: 100vw; height: 100vh;
  background: rgba(0,0,0,0.7); z-index: 9999; justify-content: center; align-items: center;
}}
#mermaid-modal.active {{ display: flex; }}
#mermaid-modal .modal-content {{
  background: #fff; padding: 20px; border-radius: 8px; max-width: 95vw; max-height: 95vh;
  overflow: auto;
}}
#mermaid-modal .close-btn {{
  position: absolute; top: 16px; right: 24px; color: #fff; font-size: 32px;
  cursor: pointer; font-weight: bold;
}}
</style>
</head>
<body>
<div class="mermaid" onclick="openModal()" title="点击放大">
{safe_code}
</div>
<div id="mermaid-modal" onclick="closeModal(event)">
  <span class="close-btn" onclick="closeModal(event)">&times;</span>
  <div class="modal-content" id="modal-body"></div>
</div>
<script>
function openModal() {{
  var modal = document.getElementById('mermaid-modal');
  var body = document.getElementById('modal-body');
  body.innerHTML = '<div class="mermaid">' + document.querySelector('.mermaid').textContent + '</div>';
  mermaid.init(undefined, body.querySelector('.mermaid'));
  modal.classList.add('active');
}}
function closeModal(e) {{
  if (e.target === document.getElementById('mermaid-modal') || e.target.classList.contains('close-btn')) {{
    document.getElementById('mermaid-modal').classList.remove('active');
  }}
}}
</script>
</body>
</html>"""
    components.html(html, height=height, scrolling=False)


def _run_multi_agent_pipeline(product_idea: str, supplementary_info: str) -> None:
    st.divider()

    # Step 1: RAG retrieval
    with st.status("🔍 正在检索相关历史案例...", expanded=True) as status:
        retriever = init_rag()
        retrieved_context = retriever.search_as_context(product_idea)
        preview = retrieved_context[:800] + "..." if len(retrieved_context) > 800 else retrieved_context
        st.markdown(f"**检索结果**：\n\n{preview}")
        status.update(label="✅ 检索完成", state="complete")

    # Step 2: Image analysis (if images uploaded)
    image_paths = st.session_state.image_paths
    if image_paths:
        st.info(f"🖼️ 已上传 {len(image_paths)} 张参考图片，将在生成过程中进行视觉分析")

    # Step 3: Run multi-agent workflow with live CoT
    from src.workflow.multi_agent.graph import NODE_DISPLAY_NAMES

    progress_bar = st.progress(0, text="多智能体协作生成中...")
    # Estimate total nodes: entry + planner + n agents + supervisor rounds + reviewer + synthesis
    total_nodes = 12  # rough estimate

    with st.status("🤖 多 Agent 协作生成中...", expanded=True) as status_container:
        cot_log = st.empty()
        node_count = 0

        def on_node(node_name: str, node_output: dict) -> None:
            nonlocal node_count
            node_count += 1
            display_name = NODE_DISPLAY_NAMES.get(node_name, node_name)
            extra = ""
            if node_name == "node_planner":
                pt = node_output.get("planner_output", {}).get("product_type", "")
                cx = node_output.get("planner_output", {}).get("complexity", "")
                if pt:
                    extra = f" — 产品类型: {pt}, 复杂度: {cx}"
            elif node_name == "node_supervisor":
                agents = node_output.get("agents_to_call", [])
                if agents:
                    extra = f" — 调度: {' → '.join(agents)}"
            elif node_name in ("node_requirements_analyst", "node_feature_planner",
                               "node_ux_designer", "node_tech_advisor"):
                if node_output.get("_reflexion_applied"):
                    before = node_output.get("_reflexion_score_before", "?")
                    extra = f" — 自评未达标({before}), 已修正"
                else:
                    extra = " — 已完成"
            elif node_name == "node_reviewer":
                score = node_output.get("reviewer_score", "?")
                extra = f" — 评分: {score}/100"
            cot_log.markdown(
                f"| {display_name} | {extra} |\n|:---|:---|\n",
                unsafe_allow_html=False,
            )
            progress_bar.progress(min(90, node_count * 8), text=f"执行中: {display_name}")

        try:
            result = run_multi_agent_workflow(
                product_idea=product_idea,
                supplementary_info=supplementary_info,
                retrieved_context=retrieved_context,
                image_paths=image_paths,
                selected_model=st.session_state.selected_model,
                temperature=st.session_state.temperature,
                reflection_max_rounds=st.session_state.reflection_max_rounds,
                reviewer_score_threshold=st.session_state.reviewer_score_threshold,
                on_node_complete=on_node,
            )
        except Exception as e:
            progress_bar.progress(100, text="❌ 生成失败")
            st.error(f"生成失败: {e}")
            return

        status_container.update(label="✅ 多 Agent 协作完成！", state="complete")

    progress_bar.progress(80, text="生成 Markdown...")

    if result.get("error_message"):
        st.error(f"生成过程中出现错误: {result['error_message']}")
        return

    final_json = result.get("final_prd_json", {})
    if not final_json:
        st.error("生成失败：未能获得最终文档")
        return

    prd_markdown = convert_to_prd_markdown(final_json)
    result["final_prd_markdown"] = prd_markdown

    progress_bar.progress(100, text="✅ 多智能体协作完成！")
    st.session_state.workflow_result = result

    # Save to history
    product_name = final_json.get("version_record", {}).get("product_name", product_idea[:30])
    st.session_state.generation_history.append({
        "product_name": product_name,
        "product_idea": product_idea,
        "timestamp": __import__("datetime").datetime.now().isoformat(),
        "result": result,
    })
    max_hist = settings.max_history
    if len(st.session_state.generation_history) > max_hist:
        st.session_state.generation_history = st.session_state.generation_history[-max_hist:]

    st.rerun()


def _display_multi_agent_results() -> None:
    st.divider()
    st.header("📄 多智能体协作生成结果")

    result = st.session_state.workflow_result

    # Image analysis
    image_analysis = result.get("image_analysis")
    if image_analysis and image_analysis.get("images"):
        with st.expander("🖼️ 图片分析结果", expanded=False):
            for img_data in image_analysis.get("images", []):
                idx = img_data.get("image_index", "?")
                img_type = img_data.get("image_type", "未知")
                st.caption(f"**图片 {idx}** — 类型：{img_type}")
                st.markdown(f"- 视觉风格：{img_data.get('visual_style', '无')}")
                st.markdown(f"- 交互流程：{img_data.get('interaction_flow', '无')}")
                st.markdown(f"- 产品洞察：{img_data.get('product_insight', '无')}")
            if image_analysis.get("aggregated_insights"):
                insights = image_analysis["aggregated_insights"]
                st.markdown(f"**整体建议**：{insights.get('design_recommendations', '无')}")

    # Reviewer score panel
    score = result.get("reviewer_score", 0)
    scores = result.get("reviewer_scores", {})
    summary = result.get("reviewer_summary", "")
    reflection_rounds = result.get("reflection_round", 0)

    st.subheader("📊 评审官评分")
    score_col1, score_col2, score_col3 = st.columns(3)
    with score_col1:
        score_color = "green" if score >= 80 else "orange" if score >= 60 else "red"
        st.metric("综合评分", f"{score}/100", delta=f"反思 {reflection_rounds} 轮")
    with score_col2:
        if scores:
            avg_agent = sum(scores.values()) // len(scores) if scores else 0
            st.metric("Agent 均分", f"{avg_agent}/100")
    with score_col3:
        threshold = st.session_state.reviewer_score_threshold
        passed = "✅ 通过" if score >= threshold else "❌ 未通过"
        st.metric("评审结果", passed)

    if summary:
        st.info(f"**评审意见**：{summary}")

    # Per-agent scores
    if scores:
        agent_labels = {
            "requirements_analyst": "需求分析师",
            "feature_planner": "功能规划师",
            "ux_designer": "体验设计师",
            "tech_advisor": "技术顾问",
        }
        score_md = " | ".join([f"{agent_labels.get(k, k)}: **{v}**" for k, v in scores.items()])
        st.markdown(score_md)

    # Reflection history
    reflection_history = result.get("reflection_history", [])
    if reflection_history:
        with st.expander(f"🔄 反思历史（{len(reflection_history)} 轮）"):
            for entry in reflection_history:
                rnd = entry.get("round", "?")
                r_score = entry.get("score", 0)
                agents = entry.get("agents_to_revise", [])
                st.caption(f"**第 {rnd} 轮** — 评分：{r_score} | 修订Agent：{', '.join(agents) if agents else '无'}")

    st.divider()

    # Agent outputs
    agent_keys = [
        ("requirement_analysis", "需求分析师", "🔍"),
        ("feature_plan", "功能规划师", "📋"),
        ("ux_design", "体验设计师", "🎨"),
        ("tech_advice", "技术顾问", "⚙️"),
    ]
    for key, label, icon in agent_keys:
        data = result.get(key)
        if data:
            with st.expander(f"{icon} {label} 输出 — 点击展开查看", expanded=False):
                if key == "ux_design":
                    _display_mermaid_diagrams(data)
                st.json(data)

    # Final PRD
    final_json = result.get("final_prd_json", {})
    if final_json:
        with st.expander("📝 文档合成（最终 PRD JSON）", expanded=False):
            st.json(final_json)

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
                key="btn_download_md_ma",
            )
        with col2:
            try:
                from src.output.exporter import export_to_pdf

                if st.button("⬇️ 导出 PDF", use_container_width=True, key="btn_export_pdf_ma"):
                    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                        pdf_path = export_to_pdf(prd_md, tmp.name)
                        with open(pdf_path, "rb") as f:
                            st.download_button(
                                "📥 点击下载 PDF",
                                data=f,
                                file_name=filename.replace(".md", ".pdf"),
                                mime="application/pdf",
                                key="btn_download_pdf_ma",
                            )
            except Exception:
                st.caption("PDF 导出需要 weasyprint 支持")

        # Revision
        st.divider()
        st.subheader("🔧 在线修改意见")
        revision_feedback = st.text_area(
            "输入修改意见，AI 将根据意见优化 PRD：",
            placeholder="例如：增加社交分享功能、修改目标用户为大学生、将技术栈从前端改为全栈...",
            height=80,
            key="input_revision_feedback_ma",
        )
        rev_col, _ = st.columns([1, 3])
        with rev_col:
            if st.button("🔧 优化 PRD", type="primary", disabled=not revision_feedback, key="btn_revise_ma"):
                from src.workflow.multi_agent.graph import NODE_DISPLAY_NAMES as _REV_NAMES

                with st.status("🔧 正在根据修改意见优化 PRD...", expanded=True) as rev_status:
                    rev_log = st.empty()

                    def on_rev_node(node_name: str, node_output: dict) -> None:
                        display_name = _REV_NAMES.get(node_name, node_name)
                        extra = ""
                        if node_name == "node_reviewer":
                            extra = f" — 评分: {node_output.get('reviewer_score', '?')}/100"
                        elif node_name in ("node_requirements_analyst", "node_feature_planner",
                                           "node_ux_designer", "node_tech_advisor"):
                            extra = " — 已完成"
                        rev_log.markdown(
                            f"| {display_name} | {extra} |\n|:---|:---|\n",
                            unsafe_allow_html=False,
                        )

                    revised = run_multi_agent_revision(result, revision_feedback, on_node_complete=on_rev_node)
                    rev_status.update(label="✅ 修订完成！", state="complete")
                    revised_final = revised.get("final_prd_json", {})
                    if revised_final:
                        revised["final_prd_markdown"] = convert_to_prd_markdown(revised_final)
                    st.session_state.workflow_result = revised
                    st.rerun()

        revision_history = result.get("revision_history", [])
        if revision_history:
            with st.expander(f"📜 修订历史（{len(revision_history)} 次）"):
                for entry in reversed(revision_history):
                    ver = entry.get("version", "?")
                    fb = entry.get("feedback", "")
                    st.caption(f"**v{ver}** — 修改意见：{fb}")


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

    # Final PRD JSON
    final_json = result.get("final_prd_json", {})
    if final_json:
        with st.expander("📝 阶段 4：文档定稿（JSON）— 点击展开查看", expanded=False):
            st.json(final_json)
            if st.button("🔄 重新生成文档定稿", key="retry_final_prd_json"):
                with st.spinner("正在重新生成文档定稿..."):
                    new_result = run_single_stage("document_finalization", result)
                    # Re-generate markdown
                    new_final = new_result.get("final_prd_json", {})
                    if new_final:
                        new_result["final_prd_markdown"] = convert_to_prd_markdown(new_final)
                    st.session_state.workflow_result = new_result
                    st.rerun()

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
                key="btn_download_md",
            )
        with col2:
            try:
                from src.output.exporter import export_to_pdf

                if st.button("⬇️ 导出 PDF", use_container_width=True, key="btn_export_pdf"):
                    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                        pdf_path = export_to_pdf(prd_md, tmp.name)
                        with open(pdf_path, "rb") as f:
                            st.download_button(
                                "📥 点击下载 PDF",
                                data=f,
                                file_name=filename.replace(".md", ".pdf"),
                                mime="application/pdf",
                                key="btn_download_pdf",
                            )
            except Exception:
                st.caption("PDF 导出需要 weasyprint 支持")

        # Revision
        st.divider()
        st.subheader("🔧 在线修改意见")
        revision_feedback = st.text_area(
            "输入修改意见，AI 将根据意见优化 PRD：",
            placeholder="例如：增加社交分享功能、修改目标用户为大学生、将技术栈从前端改为全栈...",
            height=80,
            key="input_revision_feedback",
        )
        rev_col, _ = st.columns([1, 3])
        with rev_col:
            if st.button("🔧 优化 PRD", type="primary", disabled=not revision_feedback, key="btn_revise"):
                with st.spinner("正在根据修改意见优化 PRD..."):
                    revised = run_revision(result, revision_feedback)
                    revised_final = revised.get("final_prd_json", {})
                    if revised_final:
                        revised["final_prd_markdown"] = convert_to_prd_markdown(revised_final)
                    st.session_state.workflow_result = revised
                    st.rerun()

        # Revision history
        revision_history = result.get("revision_history", [])
        if revision_history:
            with st.expander(f"📜 修订历史（{len(revision_history)} 次）"):
                for entry in reversed(revision_history):
                    ver = entry.get("version", "?")
                    fb = entry.get("feedback", "")
                    st.caption(f"**v{ver}** — 修改意见：{fb}")


if __name__ == "__main__":
    main()
