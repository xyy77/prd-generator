import time
from collections.abc import Callable

from openai import OpenAI
from openai.types.chat import ChatCompletion

from src.config import Settings, settings
from src.utils.exceptions import LLMError
from src.utils.logger import get_logger

logger = get_logger(__name__)


class LLMClient:
    def __init__(self, config: Settings | None = None):
        self.config = config or settings
        self.client = OpenAI(
            api_key=self.config.deepseek_api_key,
            base_url=self.config.deepseek_base_url,
            timeout=self.config.request_timeout,
        )
        self.model = self.config.deepseek_model
        self.temperature = self.config.llm_temperature
        self.max_retries = self.config.max_retries

    def chat(
        self,
        messages: list[dict],
        temperature: float | None = None,
        response_format: dict | None = None,
    ) -> str:
        temp = temperature if temperature is not None else self.temperature
        kwargs: dict = dict(
            model=self.model,
            messages=messages,
            temperature=temp,
        )
        if response_format is not None:
            kwargs["response_format"] = response_format

        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                logger.debug("LLM call attempt %d/%d", attempt + 1, self.max_retries + 1)
                completion: ChatCompletion = self.client.chat.completions.create(**kwargs)
                content = completion.choices[0].message.content or ""
                logger.debug("LLM returned %d chars", len(content))
                return content
            except Exception as e:
                last_error = e
                logger.warning("LLM call failed (attempt %d): %s", attempt + 1, e)
                if attempt < self.max_retries:
                    wait = 2 ** attempt
                    time.sleep(wait)

        raise LLMError(f"LLM call failed after {self.max_retries + 1} attempts: {last_error}")

    def chat_with_json_mode(self, messages: list[dict]) -> str:
        return self.chat(messages, response_format={"type": "json_object"})

    def stream_chat(
        self,
        messages: list[dict],
        on_token: Callable[[str], None] | None = None,
        temperature: float | None = None,
    ) -> str:
        temp = temperature if temperature is not None else self.temperature
        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temp,
                stream=True,
            )
            collected: list[str] = []
            for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                if delta:
                    collected.append(delta)
                    if on_token:
                        on_token(delta)
            return "".join(collected)
        except Exception as e:
            raise LLMError(f"LLM streaming failed: {e}") from e


_llm_client: LLMClient | None = None


def get_llm_client() -> LLMClient:
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client
