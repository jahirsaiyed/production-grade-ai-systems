import logging

from tenacity import retry, stop_after_attempt, wait_fixed

logger = logging.getLogger(__name__)

_MOCK_ANSWER = (
    "This is a mock answer. Set OPENAI_API_KEY in .env to call a real model."
)


class LlmCallError(RuntimeError):
    """Raised when the upstream LLM call fails after retries."""


class LlmClient:
    def __init__(self, api_key: str | None):
        self._api_key = api_key

    @property
    def is_mock(self) -> bool:
        return not self._api_key

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(0.1), reraise=True)
    def complete(self, prompt: str) -> str:
        if self.is_mock:
            return _MOCK_ANSWER
        return self._call_real_api(prompt)

    def _call_real_api(self, prompt: str) -> str:
        from openai import OpenAI

        try:
            client = OpenAI(api_key=self._api_key)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            logger.warning("llm call failed, will retry if attempts remain")
            raise LlmCallError(str(exc)) from exc
