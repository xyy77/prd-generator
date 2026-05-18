# 智能 PRD 自动生成平台

> 输入一句话产品想法，AI 自动生成符合大厂规范的完整 PRD 文档。

[![Hugging Face Spaces](https://img.shields.io/badge/Hugging%20Face-Spaces-blue?logo=huggingface)](https://huggingface.co/spaces/tina-su/prd-generator)
[![GitHub](https://img.shields.io/badge/GitHub-Repo-black?logo=github)](https://github.com/xyy77/prd-generator)

**在线体验**：https://huggingface.co/spaces/tina-su/prd-generator

基于 **RAG + LangGraph** 构建，结合历史优秀 PRD 案例知识库，通过 3 阶段并行工作流生成规范的产品需求文档。

## 功能特性

- **智能生成**：输入产品想法，自动生成包含 9 大模块的完整 PRD（版本记录、背景目标、用户画像、功能需求、非功能需求、技术架构、数据分析、风险评估、附录）
- **RAG 知识库**：基于 ChromaDB + BGE 嵌入的历史 PRD 案例语义检索，确保生成质量对标优秀案例
- **并行工作流**：需求分析 + 架构设计并行执行，大幅缩短生成时间
- **在线修订**：对生成结果提出修改意见，AI 自动优化并保留修订历史
- **Mermaid 流程图**：自动生成用户流程、状态机等可视化图表
- **文件上传**：支持上传 MD/PDF/DOCX 扩充知识库
- **多格式导出**：Markdown / PDF 一键下载

## 快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone https://github.com/YOUR_USERNAME/prd-generator.git
cd prd-generator

# 创建虚拟环境
python -m venv .venv

# 激活环境
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置 API Key

```bash
cp .env.example .env
```

编辑 `.env` 文件，填入你的 DeepSeek API Key：

```
DEEPSEEK_API_KEY=sk-your-key-here
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
DEEPSEEK_MODEL=deepseek-chat
```

### 3. 初始化知识库

```bash
# 将 prd_samples/ 目录下的示例文档导入知识库
python -m src.rag.store --seed prd_samples/
```

### 4. 启动应用

```bash
streamlit run app.py
```

浏览器打开 http://localhost:8501 即可使用。

## 项目结构

```
prd-generator/
├── app.py                          # Streamlit 入口
├── requirements.txt                # Python 依赖
├── pyproject.toml                  # 项目元数据
├── .env.example                    # 环境变量模板
│
├── src/
│   ├── config.py                   # pydantic-settings 配置
│   ├── rag/                        # RAG 知识库模块
│   │   ├── loader.py               # MD/PDF/DOCX 加载器
│   │   ├── chunker.py              # 文本切割
│   │   ├── embedder.py             # BGE 嵌入服务
│   │   ├── store.py                # ChromaDB 存储
│   │   └── retriever.py            # 语义检索
│   ├── workflow/                   # LangGraph 工作流
│   │   ├── state.py                # WorkflowState 定义
│   │   ├── graph.py                # StateGraph 构建+路由
│   │   ├── node_utils.py           # JSON 解析等工具
│   │   └── nodes/                  # 各阶段节点
│   │       ├── parallel_analysis.py    # 需求+架构并行
│   │       ├── process_flow.py         # 流程梳理
│   │       ├── document_finalization.py # 文档定稿
│   │       └── prd_revision.py         # 在线修订
│   ├── prompts/                    # Prompt 模板
│   │   ├── templates.py            # 提示词模板
│   │   └── manager.py              # Prompt 管理器
│   ├── output/                     # 输出处理
│   │   ├── validator.py            # JSON Schema 校验
│   │   ├── json_to_markdown.py     # JSON → Markdown
│   │   └── exporter.py             # 导出 MD/PDF
│   └── utils/                      # 通用工具
│       ├── llm_client.py           # DeepSeek API 客户端
│       ├── logger.py               # 日志配置
│       └── exceptions.py           # 自定义异常
│
├── prd_samples/                    # 初始知识库样本
├── data/
│   ├── chroma_db/                  # ChromaDB 持久化
│   └── uploads/                    # 用户上传文件
└── tests/                          # 测试
    ├── test_rag/
    ├── test_workflow/
    ├── test_prompts/
    ├── test_output/
    └── test_integration/
```

## 运行测试

```bash
# 运行所有非集成测试
pytest tests/ -v -m "not integration"

# 运行集成测试（需要 API Key）
pytest tests/ -v -m integration
```

## 技术栈

- **LLM**: DeepSeek V4 Pro / DeepSeek Chat（OpenAI 兼容接口）
- **工作流**: LangGraph StateGraph，3 阶段并行管线
- **RAG**: ChromaDB + BAAI/bge-base-zh-v1.5 嵌入 + 语义检索
- **UI**: Streamlit
- **文档处理**: PyMuPDF (PDF) + python-docx (DOCX) + markdown
- **配置**: pydantic-settings

## License

MIT
