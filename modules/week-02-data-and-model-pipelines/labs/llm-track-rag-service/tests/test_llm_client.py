from unittest.mock import patch

from app.adapters.llm_client import LlmClient


def test_complete_returns_mock_answer_when_no_api_key():
    client = LlmClient(api_key=None)
    assert client.is_mock is True
    assert "mock answer" in client.complete("anything").lower()


def test_complete_calls_real_api_when_key_present():
    client = LlmClient(api_key="fake-key")
    with patch.object(
        client, "_call_real_api", return_value="real answer"
    ) as mock_call:
        result = client.complete("hello")
    assert result == "real answer"
    mock_call.assert_called_once_with("hello")
