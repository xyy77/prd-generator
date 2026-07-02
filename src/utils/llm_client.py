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
        model: str | None = None,
    ) -> str:
        temp = temperature if temperature is not None else self.temperature
        kwargs: dict = dict(
            model=model or self.model,
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

    def chat_with_json_mode(self, messages: list[dict], model: str | None = None) -> str:
        return self.chat(messages, response_format={"type": "json_object"}, model=model)

    def stream_chat(
        self,
        messages: list[dict],
        on_token: Callable[[str], None] | None = None,
        temperature: float | None = None,
        model: str | None = None,
    ) -> str:
        temp = temperature if temperature is not None else self.temperature
        try:
            stream = self.client.chat.completions.create(
                model=model or self.model,
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


# ── Multi-Provider LLM Client ────────────────────────────────────


class MultiProviderLLMClient:
    """LLM client with automatic fallback across multiple providers.

    Fallback chain (by priority):
      1. DeepSeek (deepseek-chat) — lowest cost
      2. 阿里云百炼 (qwen-plus) — DashScope OpenAI-compatible
      3. 智谱 (glm-4-flash) — Zhipu OpenAI-compatible

    All three use the OpenAI SDK — no extra dependencies needed.

    Fallback triggers: timeout, 429 rate-limit, 5xx server errors.
    Does NOT fallback on 4xx (API key invalid, bad request, etc.).
    """

    def __init__(self, config: Settings | None = None):
        self.config = config or settings
        self.temperature = self.config.llm_temperature
        self.max_retries = self.config.max_retries
        self._fallback_log: list[dict] = []

        self._providers: list[dict] = []
        # Provider 1: DeepSeek (always available — primary)
        if self.config.deepseek_api_key and "placeholder" not in self.config.deepseek_api_key:
            self._providers.append({
                "name": "deepseek",
                "client": OpenAI(
                    api_key=self.config.deepseek_api_key,
                    base_url=self.config.deepseek_base_url,
                    timeout=self.config.request_timeout,
                ),
                "default_model": self.config.deepseek_model,
                "priority": 1,
            })

        # Provider 2: Bailian (DashScope) — reuses dashscope_api_key
        if self.config.dashscope_api_key:
            self._providers.append({
                "name": "bailian",
                "client": OpenAI(
                    api_key=self.config.dashscope_api_key,
                    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                    timeout=self.config.request_timeout,
                ),
                "default_model": self.config.dashscope_text_model,
                "priority": 2,
            })

        # Provider 3: Zhipu (GLM) — reuses zhipu_api_key
        if self.config.zhipu_api_key:
            self._providers.append({
                "name": "zhipu",
                "client": OpenAI(
                    api_key=self.config.zhipu_api_key,
                    base_url="https://open.bigmodel.cn/api/paas/v4",
                    timeout=self.config.request_timeout,
                ),
                "default_model": self.config.zhipu_text_model,
                "priority": 3,
            })

        self._providers.sort(key=lambda p: p["priority"])

    @property
    def fallback_history(self) -> list[dict]:
        return list(self._fallback_log)

    @property
    def available_providers(self) -> list[str]:
        return [p["name"] for p in self._providers]

    def _is_fallback_error(self, err: Exception) -> bool:
        """Check if an error should trigger provider fallback."""
        import httpx
        if isinstance(err, httpx.TimeoutException):
            return True
        if isinstance(err, httpx.ConnectError):
            return True
        if isinstance(err, httpx.RemoteProtocolError):
            return True
        if hasattr(err, "status_code"):
            code = getattr(err, "status_code", 0)
            if 400 <= code < 500:
                if code in (429, 408):
                    return True  # rate limit or timeout
                return False  # 4xx: bad request, auth error, etc.
            if code >= 500:
                return True
        err_msg = str(err).lower()
        if any(kw in err_msg for kw in ("timeout", "rate limit", "too many requests",
                                          "server error", "internal error", "connection",
                                          "service unavailable")):
            return True
        return False

    def chat(
        self,
        messages: list[dict],
        temperature: float | None = None,
        response_format: dict | None = None,
        model: str | None = None,
    ) -> str:
        temp = temperature if temperature is not None else self.temperature
        last_error: Exception | None = None

        for provider in self._providers:
            provider_name = provider["name"]
            client = provider["client"]
            use_model = model or provider["default_model"]

            try:
                kwargs: dict = dict(
                    model=use_model,
                    messages=messages,
                    temperature=temp,
                )
                if response_format is not None:
                    kwargs["response_format"] = response_format

                completion = client.chat.completions.create(**kwargs)
                content = completion.choices[0].message.content or ""
                logger.debug("MultiProvider [%s] returned %d chars", provider_name, len(content))
                self._fallback_log.append({
                    "provider": provider_name, "model": use_model,
                    "status": "success", "chars": len(content),
                })
                return content

            except Exception as e:
                should_fallback = self._is_fallback_error(e)
                logger.warning(
                    "MultiProvider [%s] failed (fallback=%s): %s",
                    provider_name, should_fallback, e,
                )
                last_error = e
                if not should_fallback:
                    raise  # 4xx auth error: don't fallback

        raise LLMError(
            f"All providers exhausted ({len(self._providers)} attempted). "
            f"Last error: {last_error}"
        )

    def chat_with_json_mode(self, messages: list[dict], model: str | None = None) -> str:
        return self.chat(messages, response_format={"type": "json_object"}, model=model)

    def chat_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        model: str | None = None,
        temperature: float | None = None,
        force_tool: bool = True,
    ) -> dict:
        """Send a chat completion with function calling (tool_choice).

        Returns the parsed tool call arguments dict, or falls back to
        JSON Mode parsing if the provider doesn't return a tool call.

        When force_tool=False, the LLM may choose not to call any tool.
        In that case the returned dict has ``_is_tool_call`` set to False.
        """
        temp = temperature if temperature is not None else self.temperature
        last_error: Exception | None = None

        for provider in self._providers:
            provider_name = provider["name"]
            client = provider["client"]
            use_model = model or provider["default_model"]

            try:
                kwargs: dict = dict(
                    model=use_model,
                    messages=messages,
                    temperature=temp,
                    tools=tools,
                )
                if force_tool:
                    kwargs["tool_choice"] = {"type": "function", "function": {"name": tools[0]["function"]["name"]}}

                completion = client.chat.completions.create(**kwargs)

                msg = completion.choices[0].message
                if msg.tool_calls:
                    tc = msg.tool_calls[0]
                    raw_args = tc.function.arguments
                    parsed = json.loads(raw_args)
                    parsed["_is_tool_call"] = True
                    parsed["_tool_name"] = tc.function.name
                    logger.debug("MultiProvider [%s] tool_call '%s' OK, args: %s keys",
                                 provider_name, tc.function.name, list(parsed.keys()))
                    self._fallback_log.append({
                        "provider": provider_name, "model": use_model,
                        "status": "tool_call", "tool_name": tc.function.name,
                    })
                    return parsed

                # No tool call returned — parse content as JSON
                content = msg.content or ""
                if content.strip():
                    logger.debug("MultiProvider [%s] no tool_call, parsing content as JSON", provider_name)
                    self._fallback_log.append({
                        "provider": provider_name, "model": use_model,
                        "status": "content", "chars": len(content),
                    })
                    parsed = json.loads(content) if content.strip().startswith("{") else {}
                    if isinstance(parsed, dict):
                        parsed["_is_tool_call"] = False
                    return parsed

                raise LLMError(f"Provider {provider_name} returned empty response with no tool call")

            except json.JSONDecodeError:
                last_error = Exception(f"Provider {provider_name} returned non-JSON content")
                logger.warning("MultiProvider [%s] content parse failed", provider_name)
            except Exception as e:
                should_fallback = self._is_fallback_error(e)
                logger.warning(
                    "MultiProvider [%s] tool call failed (fallback=%s): %s",
                    provider_name, should_fallback, e,
                )
                last_error = e
                if not should_fallback:
                    raise

        raise LLMError(
            f"All providers exhausted for tool call ({len(self._providers)} attempted). "
            f"Last error: {last_error}"
        )

    def stream_chat(
        self,
        messages: list[dict],
        on_token: Callable[[str], None] | None = None,
        temperature: float | None = None,
        model: str | None = None,
    ) -> str:
        temp = temperature if temperature is not None else self.temperature
        last_error: Exception | None = None

        for provider in self._providers:
            provider_name = provider["name"]
            client = provider["client"]
            use_model = model or provider["default_model"]

            try:
                stream = client.chat.completions.create(
                    model=use_model,
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
                result = "".join(collected)
                self._fallback_log.append({
                    "provider": provider_name, "model": use_model,
                    "status": "stream_ok", "chars": len(result),
                })
                return result

            except Exception as e:
                should_fallback = self._is_fallback_error(e)
                logger.warning(
                    "MultiProvider [%s] stream failed (fallback=%s): %s",
                    provider_name, should_fallback, e,
                )
                last_error = e
                if not should_fallback:
                    raise

        raise LLMError(
            f"All providers stream exhausted. Last error: {last_error}"
        )


_multi_llm_client: MultiProviderLLMClient | None = None


def get_multi_llm_client() -> MultiProviderLLMClient:
    global _multi_llm_client
    if _multi_llm_client is None:
        _multi_llm_client = MultiProviderLLMClient()
    return _multi_llm_client
