"""
Unit tests for InertialAIEmbeddingFunction - specifically the
static methods like name, default_space, and supported_spaces,
ensuring they return the expected values.
"""

from __future__ import annotations

import pytest

from inertialai_chroma import InertialAIEmbeddingFunction

pytestmark = pytest.mark.usefixtures("api_key")


def test_name() -> None:
    assert InertialAIEmbeddingFunction.name() == "inertialai"


def test_default_space() -> None:
    ef = InertialAIEmbeddingFunction()
    assert ef.default_space() == "cosine"


def test_supported_spaces() -> None:
    ef = InertialAIEmbeddingFunction()
    assert set(ef.supported_spaces()) == {"cosine", "l2", "ip"}
