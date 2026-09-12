from unittest.mock import patch

from app.adapters.llm_client import LlmClient
from judge import judge_answer


def test_judge_answer_reports_not_meaningful_in_mock_mode():
    client = LlmClient(api_key=None)
    result = judge_answer("q", "hr-policy.md", "some answer", client)
    assert result["is_meaningful"] is False


def test_judge_answer_reports_meaningful_when_using_a_real_client():
    client = LlmClient(api_key="fake-key")
    with patch.object(client, "_call_real_api", return_value="YES") as mock_call:
        result = judge_answer("q", "hr-policy.md", "some answer", client)
    assert result["is_meaningful"] is True
    assert result["verdict"] == "YES"
    mock_call.assert_called_once()
