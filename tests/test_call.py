"""
Unit tests for InertialAIEmbeddingFunction - specifically the __call__ method
and how it handles different input formats and API interactions.
"""

from __future__ import annotations

import json

import httpx
import numpy as np
import pytest
import respx

from inertialai_chroma import InertialAIEmbeddingFunction

pytestmark = pytest.mark.usefixtures("api_key")

API_URL = "https://inertialai.com/api/v1/embeddings"


def test_empty_input_does_not_call_api(mock_api: respx.MockRouter) -> None:
    """
    Chroma raises for empty input, but our guard must not make an HTTP call first.
    """
    route = mock_api.post(API_URL).mock(
        return_value=httpx.Response(
            201,
            json={
                "object": "list",
                "model": "inertial-embed-alpha",
                "data": [],
                "usage": {"prompt_tokens": 0, "total_tokens": 0},
                "create_time": "2024-01-01T00:00:00Z",
            },
        )
    )
    ef = InertialAIEmbeddingFunction()
    with pytest.raises(ValueError):
        ef([])
    assert not route.called


def test_plain_text_wrapped_as_text_dict(mock_api: respx.MockRouter) -> None:
    route = mock_api.post(API_URL).mock(
        return_value=httpx.Response(
            201,
            json={
                "object": "list",
                "model": "inertial-embed-alpha",
                "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2]}],
                "usage": {"prompt_tokens": 5, "total_tokens": 5},
                "create_time": "2024-01-01T00:00:00Z",
            },
        )
    )
    ef = InertialAIEmbeddingFunction()
    ef(["hello world"])

    request_body = json.loads(route.calls[0].request.content)
    assert request_body["input"] == [{"text": "hello world"}]


def test_json_multimodal_dict_passed_directly(mock_api: respx.MockRouter) -> None:
    route = mock_api.post(API_URL).mock(
        return_value=httpx.Response(
            201,
            json={
                "object": "list",
                "model": "inertial-embed-alpha",
                "data": [{"object": "embedding", "index": 0, "embedding": [0.3, 0.4]}],
                "usage": {"prompt_tokens": 10, "total_tokens": 10},
                "create_time": "2024-01-01T00:00:00Z",
            },
        )
    )
    ef = InertialAIEmbeddingFunction()
    doc = json.dumps({"text": "Energy spike", "time_series": [[1.2, 1.5, 1.8]]})
    ef([doc])

    request_body = json.loads(route.calls[0].request.content)
    assert request_body["input"] == [{"text": "Energy spike", "time_series": [[1.2, 1.5, 1.8]]}]


def test_mixed_plain_and_multimodal_documents(mock_api: respx.MockRouter) -> None:
    route = mock_api.post(API_URL).mock(
        return_value=httpx.Response(
            201,
            json={
                "object": "list",
                "model": "inertial-embed-alpha",
                "data": [
                    {"object": "embedding", "index": 0, "embedding": [0.1, 0.2]},
                    {"object": "embedding", "index": 1, "embedding": [0.3, 0.4]},
                ],
                "usage": {"prompt_tokens": 15, "total_tokens": 15},
                "create_time": "2024-01-01T00:00:00Z",
            },
        )
    )
    ef = InertialAIEmbeddingFunction()
    plain = "plain text"
    multimodal = json.dumps({"time_series": [[0.5, 0.6]]})
    ef([plain, multimodal])

    request_body = json.loads(route.calls[0].request.content)
    assert request_body["input"][0] == {"text": "plain text"}
    assert request_body["input"][1] == {"time_series": [[0.5, 0.6]]}


def test_returns_numpy_float32_arrays(single_embedding_route: respx.Route) -> None:
    ef = InertialAIEmbeddingFunction()
    result = ef(["test"])
    assert len(result) == 1
    assert isinstance(result[0], np.ndarray)
    assert result[0].dtype == np.float32
    np.testing.assert_array_almost_equal(result[0], [0.1, 0.2, 0.3])


def test_dimensions_sent_in_request_body(mock_api: respx.MockRouter) -> None:
    route = mock_api.post(API_URL).mock(
        return_value=httpx.Response(
            201,
            json={
                "object": "list",
                "model": "inertial-embed-alpha",
                "data": [{"object": "embedding", "index": 0, "embedding": [0.1]}],
                "usage": {"prompt_tokens": 5, "total_tokens": 5},
                "create_time": "2024-01-01T00:00:00Z",
            },
        )
    )
    ef = InertialAIEmbeddingFunction(dimensions=128)
    ef(["test"])

    request_body = json.loads(route.calls[0].request.content)
    assert request_body["dimensions"] == 128


def test_dimensions_not_sent_when_none(mock_api: respx.MockRouter) -> None:
    route = mock_api.post(API_URL).mock(
        return_value=httpx.Response(
            201,
            json={
                "object": "list",
                "model": "inertial-embed-alpha",
                "data": [{"object": "embedding", "index": 0, "embedding": [0.1]}],
                "usage": {"prompt_tokens": 5, "total_tokens": 5},
                "create_time": "2024-01-01T00:00:00Z",
            },
        )
    )
    ef = InertialAIEmbeddingFunction()
    ef(["test"])

    request_body = json.loads(route.calls[0].request.content)
    assert "dimensions" not in request_body


def test_bearer_token_sent_in_header(api_key: str, mock_api: respx.MockRouter) -> None:
    route = mock_api.post(API_URL).mock(
        return_value=httpx.Response(
            201,
            json={
                "object": "list",
                "model": "inertial-embed-alpha",
                "data": [{"object": "embedding", "index": 0, "embedding": [0.1]}],
                "usage": {"prompt_tokens": 5, "total_tokens": 5},
                "create_time": "2024-01-01T00:00:00Z",
            },
        )
    )
    ef = InertialAIEmbeddingFunction()
    ef(["test"])

    auth_header = route.calls[0].request.headers.get("authorization", "")
    assert auth_header == f"Bearer {api_key}"


def test_http_error_propagates(mock_api: respx.MockRouter) -> None:
    mock_api.post(API_URL).mock(return_value=httpx.Response(401))
    ef = InertialAIEmbeddingFunction()
    with pytest.raises(httpx.HTTPStatusError):
        ef(["test"])


def test_json_array_not_treated_as_multimodal(mock_api: respx.MockRouter) -> None:
    """
    A JSON array (not a dict) should be treated as plain text.
    """
    route = mock_api.post(API_URL).mock(
        return_value=httpx.Response(
            201,
            json={
                "object": "list",
                "model": "inertial-embed-alpha",
                "data": [{"object": "embedding", "index": 0, "embedding": [0.1]}],
                "usage": {"prompt_tokens": 5, "total_tokens": 5},
                "create_time": "2024-01-01T00:00:00Z",
            },
        )
    )
    ef = InertialAIEmbeddingFunction()
    ef(["[1, 2, 3]"])

    request_body = json.loads(route.calls[0].request.content)
    assert request_body["input"] == [{"text": "[1, 2, 3]"}]
