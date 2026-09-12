from unittest.mock import patch

import numpy as np

from app.adapters.embeddings import EmbeddingClient


def test_embed_is_deterministic_in_mock_mode():
    client = EmbeddingClient(api_key=None)
    assert client.is_mock is True
    vec1 = client.embed("hello world")
    vec2 = client.embed("hello world")
    assert np.array_equal(vec1, vec2)


def test_embed_produces_different_vectors_for_different_text():
    client = EmbeddingClient(api_key=None)
    vec1 = client.embed("hello world")
    vec2 = client.embed("completely different sentence")
    assert not np.array_equal(vec1, vec2)


def test_embed_gives_shared_words_more_overlap_than_unrelated_text():
    client = EmbeddingClient(api_key=None)
    vec_a = client.embed("vacation policy days")
    vec_b = client.embed("vacation policy allowance")
    vec_c = client.embed("deploy rollback incident")

    def cosine(a, b):
        return float(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b))

    assert cosine(vec_a, vec_b) > cosine(vec_a, vec_c)


def test_embed_calls_real_api_when_key_present():
    client = EmbeddingClient(api_key="fake-key")
    fake_vector = np.array([0.1, 0.2])
    with patch.object(
        client, "_call_real_api", return_value=fake_vector
    ) as mock_call:
        result = client.embed("hello")
    assert np.array_equal(result, fake_vector)
    mock_call.assert_called_once_with("hello")
