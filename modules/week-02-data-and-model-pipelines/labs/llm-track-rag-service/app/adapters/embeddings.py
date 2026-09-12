import hashlib
import logging

import numpy as np
from tenacity import retry, stop_after_attempt, wait_fixed

logger = logging.getLogger(__name__)

MOCK_EMBEDDING_DIM = 64


class EmbeddingCallError(RuntimeError):
    """Raised when the upstream embeddings call fails after retries."""


def _mock_embed(text: str) -> np.ndarray:
    vector = np.zeros(MOCK_EMBEDDING_DIM)
    for word in text.lower().split():
        digest = hashlib.sha256(word.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % MOCK_EMBEDDING_DIM
        vector[index] += 1.0
    return vector


class EmbeddingClient:
    def __init__(self, api_key: str | None):
        self._api_key = api_key

    @property
    def is_mock(self) -> bool:
        return not self._api_key

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(0.1), reraise=True)
    def embed(self, text: str) -> np.ndarray:
        if self.is_mock:
            return _mock_embed(text)
        return self._call_real_api(text)

    def _call_real_api(self, text: str) -> np.ndarray:
        from openai import OpenAI

        try:
            client = OpenAI(api_key=self._api_key)
            response = client.embeddings.create(
                model="text-embedding-3-small", input=text
            )
            return np.array(response.data[0].embedding)
        except Exception as exc:
            logger.warning("embedding call failed, will retry if attempts remain")
            raise EmbeddingCallError(str(exc)) from exc
