"""
Shared fixtures for inertialai-chroma unit tests.
"""

from __future__ import annotations

from collections.abc import Generator
from typing import Any

import httpx
import pytest
import respx

FAKE_API_KEY = "test-key-abc123"
API_URL = "https://inertialai.com/api/v1/embeddings"


def _embedding_response(*embeddings: list[float]) -> dict[str, Any]:
    """
    Build a minimal API response payload for the given embedding vectors.
    """
    return {
        "object": "list",
        "model": "inertial-embed-alpha",
        "data": [
            {"object": "embedding", "index": i, "embedding": emb}
            for i, emb in enumerate(embeddings)
        ],
        "usage": {"prompt_tokens": 10, "total_tokens": 15},
        "create_time": "2024-01-01T00:00:00.000000Z",
    }


@pytest.fixture()
def api_key(monkeypatch: pytest.MonkeyPatch) -> str:
    """
    Set INERTIALAI_API_KEY in the environment and return its value.
    """
    monkeypatch.setenv("INERTIALAI_API_KEY", FAKE_API_KEY)
    return FAKE_API_KEY


@pytest.fixture()
def mock_api() -> Generator[respx.MockRouter, None, None]:
    """
    Context manager that mocks the InertialAI embeddings endpoint.
    """
    with respx.mock(assert_all_called=False) as mock:
        yield mock


@pytest.fixture()
def single_embedding_route(mock_api: respx.MockRouter) -> respx.Route:
    """
    Mock route returning a single embedding of [0.1, 0.2, 0.3].
    """
    return mock_api.post(API_URL).mock(
        return_value=httpx.Response(201, json=_embedding_response([0.1, 0.2, 0.3]))
    )


@pytest.fixture()
def two_embedding_route(mock_api: respx.MockRouter) -> respx.Route:
    """
    Mock route returning two embeddings.
    """
    return mock_api.post(API_URL).mock(
        return_value=httpx.Response(
            201,
            json=_embedding_response([0.1, 0.2, 0.3], [0.4, 0.5, 0.6]),
        )
    )
