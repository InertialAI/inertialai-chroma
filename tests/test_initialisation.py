"""
Unit tests for InertialAIEmbeddingFunction - specifically the __init__ method
and how it handles API key configuration, model name, dimensions, and timeout settings.
"""

from __future__ import annotations

import warnings

import pytest

from inertialai_chroma import InertialAIEmbeddingFunction

pytestmark = pytest.mark.usefixtures("api_key")


def test_reads_api_key_from_env(api_key: str) -> None:
    ef = InertialAIEmbeddingFunction()
    assert ef.api_key == api_key


def test_direct_api_key_raises_deprecation_warning() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ef = InertialAIEmbeddingFunction(api_key="direct-key")
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)
    assert ef.api_key == "direct-key"


def test_missing_env_var_raises_value_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("INERTIALAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="INERTIALAI_API_KEY"):
        InertialAIEmbeddingFunction()


def test_custom_env_var_name(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MY_CUSTOM_KEY", "secret")
    ef = InertialAIEmbeddingFunction(api_key_env_var="MY_CUSTOM_KEY")
    assert ef.api_key == "secret"
    assert ef.api_key_env_var == "MY_CUSTOM_KEY"


def test_default_model_name() -> None:
    ef = InertialAIEmbeddingFunction()
    assert ef.model_name == "inertial-embed-alpha"


def test_custom_model_name() -> None:
    ef = InertialAIEmbeddingFunction(model_name="other-model")
    assert ef.model_name == "other-model"


def test_dimensions_none_by_default() -> None:
    ef = InertialAIEmbeddingFunction()
    assert ef.dimensions is None


def test_dimensions_stored() -> None:
    ef = InertialAIEmbeddingFunction(dimensions=512)
    assert ef.dimensions == 512


def test_timeout_default() -> None:
    ef = InertialAIEmbeddingFunction()
    assert ef.timeout == 60.0


def test_timeout_custom() -> None:
    ef = InertialAIEmbeddingFunction(timeout=120.0)
    assert ef.timeout == 120.0
