import base64
import json
import mimetypes
from pathlib import Path

from src.config import Settings, settings
from src.utils.exceptions import LLMError
from src.utils.logger import get_logger

logger = get_logger(__name__)

SYSTEM_PROMPT = """你是一位产品分析专家，擅长从截图、草图和流程图中提取产品设计信息。
请分析上传的图片，提取以下内容：
1. 界面元素和布局：识别UI组件（按钮、输入框、导航栏、卡片等）和它们的排列方式
2. 交互流程：从图片中推断用户操作步骤和页面跳转逻辑
3. 视觉风格：整体设计风格、色彩倾向、组件风格
4. 产品洞察：图片传达的产品功能意图

请以JSON格式输出，key使用英文，value使用中文。"""


class VisionClient:
    def __init__(self, config: Settings | None = None):
        self.config = config or settings
        self.provider = self.config.vision_provider

    def analyze_images(
        self,
        image_paths: list[str],
        product_idea: str,
    ) -> dict:
        if not image_paths:
            return {}

        valid_paths = [p for p in image_paths if Path(p).exists()]
        if not valid_paths:
            logger.warning("No valid image paths found")
            return {}

        user_prompt = f"产品想法: {product_idea}\n请分析以下图片，提取界面元素、交互流程、视觉风格和产品洞察。"

        try:
            if self.provider == "qwen":
                return self._analyze_qwen(valid_paths, user_prompt)
            elif self.provider == "zhipu":
                return self._analyze_zhipu(valid_paths, user_prompt)
            elif self.provider == "openai":
                return self._analyze_openai(valid_paths, user_prompt)
            elif self.provider == "anthropic":
                return self._analyze_anthropic(valid_paths, user_prompt)
            else:
                logger.warning("Vision provider '%s' not recognized, skipping image analysis", self.provider)
                return {}
        except Exception as e:
            logger.error("Vision analysis failed: %s", e)
            return {}

    def _analyze_qwen(self, image_paths: list[str], user_prompt: str) -> dict:
        try:
            import dashscope
            from dashscope import MultiModalConversation
        except ImportError:
            logger.error("dashscope not installed, install with: pip install dashscope")
            return {}

        dashscope.api_key = self.config.dashscope_api_key or ""

        content = []
        for p in image_paths:
            content.append({"image": f"file://{p}"})
        content.append({"text": SYSTEM_PROMPT + "\n\n" + user_prompt})

        messages = [{"role": "user", "content": content}]

        try:
            response = MultiModalConversation.call(
                model=self.config.vision_model,
                messages=messages,
            )
            raw = response.output.choices[0].message.content[0]["text"]
            return self._parse_vision_response(raw)
        except Exception as e:
            raise LLMError(f"Qwen-VL call failed: {e}") from e

    def _analyze_zhipu(self, image_paths: list[str], user_prompt: str) -> dict:
        try:
            from zhipuai import ZhipuAI
        except ImportError:
            logger.error("zhipuai not installed, install with: pip install zhipuai")
            return {}

        api_key = getattr(self.config, "zhipu_api_key", "") or ""
        client = ZhipuAI(api_key=api_key)

        content = [{"type": "text", "text": SYSTEM_PROMPT + "\n\n" + user_prompt}]
        for p in image_paths:
            mime_type, _ = mimetypes.guess_type(p)
            if not mime_type or not mime_type.startswith("image/"):
                mime_type = "image/png"
            with open(p, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{b64}"},
            })

        try:
            response = client.chat.completions.create(
                model=self.config.vision_model or "glm-4v-flash",
                messages=[{"role": "user", "content": content}],
                temperature=0.3,
                max_tokens=4096,
            )
            raw = response.choices[0].message.content or ""
            return self._parse_vision_response(raw)
        except Exception as e:
            raise LLMError(f"Zhipu Vision call failed: {e}") from e

    def _analyze_openai(self, image_paths: list[str], user_prompt: str) -> dict:
        from openai import OpenAI

        client = OpenAI(
            api_key=self.config.deepseek_api_key,
            base_url=self.config.deepseek_base_url,
            timeout=self.config.request_timeout,
        )

        content = [{"type": "text", "text": SYSTEM_PROMPT + "\n\n" + user_prompt}]
        for p in image_paths:
            mime_type, _ = mimetypes.guess_type(p)
            if not mime_type or not mime_type.startswith("image/"):
                mime_type = "image/png"
            with open(p, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{b64}"},
            })

        try:
            response = client.chat.completions.create(
                model=self.config.vision_model or "gpt-4o",
                messages=[{"role": "user", "content": content}],
                response_format={"type": "json_object"},
                temperature=0.3,
            )
            raw = response.choices[0].message.content or ""
            return self._parse_vision_response(raw)
        except Exception as e:
            raise LLMError(f"OpenAI Vision call failed: {e}") from e

    def _analyze_anthropic(self, image_paths: list[str], user_prompt: str) -> dict:
        try:
            import anthropic
        except ImportError:
            logger.error("anthropic not installed, install with: pip install anthropic")
            return {}

        client = anthropic.Anthropic(
            api_key=self.config.deepseek_api_key,
        )

        content = [{"type": "text", "text": SYSTEM_PROMPT + "\n\n" + user_prompt}]
        for p in image_paths:
            mime_type, _ = mimetypes.guess_type(p)
            if not mime_type or not mime_type.startswith("image/"):
                mime_type = "image/png"
            with open(p, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": mime_type, "data": b64},
            })

        try:
            response = client.messages.create(
                model=self.config.vision_model or "claude-3-opus-20240229",
                max_tokens=4096,
                messages=[{"role": "user", "content": content}],
            )
            raw = response.content[0].text if response.content else ""
            return self._parse_vision_response(raw)
        except Exception as e:
            raise LLMError(f"Anthropic Vision call failed: {e}") from e

    def _parse_vision_response(self, raw: str) -> dict:
        raw = raw.strip()
        if raw.startswith("```"):
            lines = raw.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            raw = "\n".join(lines)

        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            start = raw.find("{")
            end = raw.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(raw[start:end + 1])
                except json.JSONDecodeError:
                    pass
            logger.warning("Could not parse vision LLM JSON response")
            return {"raw_analysis": raw}
