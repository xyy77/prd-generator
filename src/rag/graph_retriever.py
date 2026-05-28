import json
import os
from pathlib import Path

from src.config import settings
from src.utils.llm_client import LLMClient
from src.utils.logger import get_logger

logger = get_logger(__name__)

GRAPH_INDEX_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "graph_index"
GRAPH_INDEX_FILE = GRAPH_INDEX_DIR / "graph.json"

EXTRACTION_PROMPT = """你是一位知识图谱工程师。请从以下PRD文档片段中提取实体和关系。

实体类型: feature_module(功能模块), user_role(用户角色), tech_component(技术组件), data_entity(数据实体)
关系类型: depends_on(依赖), contains(包含), triggers(触发), interacts_with(交互)

返回 JSON:
{
  "entities": [
    {"name": "用户登录", "type": "feature_module", "description": "..."},
    {"name": "普通用户", "type": "user_role", "description": "..."}
  ],
  "relations": [
    {"source": "用户登录", "target": "用户认证服务", "type": "depends_on", "description": "登录功能依赖认证服务"},
    {"source": "发帖功能", "target": "普通用户", "type": "interacts_with", "description": "用户可以发帖"}
  ]
}

只输出 JSON，不要包含任何其他文字。"""


class GraphRetriever:
    """Lightweight knowledge-graph retriever layered on top of vector search."""

    def __init__(self, graph_path: str | None = None):
        self.graph_path = graph_path or str(GRAPH_INDEX_FILE)
        self._graph: dict | None = None

    @property
    def graph(self) -> dict:
        if self._graph is None:
            self._graph = self._load_graph()
        return self._graph

    @property
    def is_ready(self) -> bool:
        return os.path.exists(self.graph_path)

    def _load_graph(self) -> dict:
        if not os.path.exists(self.graph_path):
            return {"entities": [], "relations": []}
        try:
            with open(self.graph_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning("Failed to load graph index: %s", e)
            return {"entities": [], "relations": []}

    def build_index(self, documents: list[str], force: bool = False) -> bool:
        """Build knowledge graph from document texts. Returns True on success."""
        if self.is_ready and not force:
            logger.info("Graph index already exists, skipping build")
            return True

        if not documents:
            logger.warning("No documents to build graph index from")
            return False

        try:
            combined = "\n\n---\n\n".join(documents)
            if len(combined) > 8000:
                combined = combined[:8000]

            client = LLMClient()
            messages = [
                {"role": "system", "content": EXTRACTION_PROMPT},
                {"role": "user", "content": f"请从以下文档中提取实体和关系：\n\n{combined}"},
            ]
            raw = client.chat_with_json_mode(messages)
            # Strip markdown fences if present
            raw = raw.strip()
            if raw.startswith("```"):
                lines = raw.split("\n")
                lines = lines[1:] if lines[0].startswith("```") else lines
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                raw = "\n".join(lines)
            graph = json.loads(raw)

            os.makedirs(os.path.dirname(self.graph_path), exist_ok=True)
            with open(self.graph_path, "w", encoding="utf-8") as f:
                json.dump(graph, f, ensure_ascii=False, indent=2)

            self._graph = graph
            logger.info(
                "Graph index built: %d entities, %d relations",
                len(graph.get("entities", [])),
                len(graph.get("relations", [])),
            )
            return True
        except Exception as e:
            logger.error("Failed to build graph index: %s", e)
            return False

    def search(self, query: str) -> str:
        """Return graph context relevant to the query. Returns empty string if no graph."""
        if not self.is_ready:
            return ""

        graph = self.graph
        entities = graph.get("entities", [])
        relations = graph.get("relations", [])

        if not entities:
            return ""

        query_lower = query.lower()
        matched_entities = [
            e for e in entities
            if any(kw in e.get("name", "") or kw in e.get("description", "")
                   for kw in query_lower.split())
        ]
        if not matched_entities:
            matched_entities = entities[:6]

        matched_names = {e["name"] for e in matched_entities}
        matched_relations = [
            r for r in relations
            if r.get("source") in matched_names or r.get("target") in matched_names
        ]

        lines: list[str] = []
        if matched_entities:
            lines.append("### 相关实体")
            for e in matched_entities[:8]:
                lines.append(f"- **{e['name']}** ({e.get('type', '')}): {e.get('description', '')}")
        if matched_relations:
            lines.append("### 实体关系")
            for r in matched_relations[:10]:
                lines.append(
                    f"- {r.get('source', '')} --[{r.get('type', '')}]--> {r.get('target', '')}"
                    + (f": {r.get('description', '')}" if r.get('description') else "")
                )

        return "\n".join(lines) if lines else ""
