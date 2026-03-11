"""
Unit tests for InertialAIEmbeddingFunction - specifically the
persistence-related methods like get_config and build_from_config,
ensuring that sensitive information is not exposed and that the
configuration can be round-tripped correctly.
"""

from __future__ import annotations

import pytest

from inertialai_chroma import InertialAIEmbeddingFunction

pytestmark = pytest.mark.usefixtures("api_key")


def test_get_config_contains_env_var_name_not_key_value(api_key: str) -> None:
    ef = InertialAIEmbeddingFunction()
    config = ef.get_config()
    assert config["api_key_env_var"] == "INERTIALAI_API_KEY"
    assert "api_key" not in config or config.get("api_key") is None
    assert api_key not in str(config)


def test_get_config_round_trip() -> None:
    ef = InertialAIEmbeddingFunction(model_name="custom-model", dimensions=256, timeout=30.0)
    config = ef.get_config()
    assert config["model_name"] == "custom-model"
    assert config["dimensions"] == 256
    assert config["timeout"] == 30.0


def test_build_from_config_reconstructs() -> None:
    config = {
        "api_key_env_var": "INERTIALAI_API_KEY",
        "model_name": "inertial-embed-alpha",
        "dimensions": None,
        "timeout": 60.0,
    }
    ef = InertialAIEmbeddingFunction.build_from_config(config)
    assert isinstance(ef, InertialAIEmbeddingFunction)
    assert ef.model_name == "inertial-embed-alpha"
    assert ef.dimensions is None
    assert ef.timeout == 60.0


def test_build_from_config_uses_default_timeout_when_absent() -> None:
    config = {
        "api_key_env_var": "INERTIALAI_API_KEY",
        "model_name": "inertial-embed-alpha",
        "dimensions": None,
    }
    ef = InertialAIEmbeddingFunction.build_from_config(config)
    assert ef.timeout == 60.0


def test_build_from_config_with_dimensions() -> None:
    config = {
        "api_key_env_var": "INERTIALAI_API_KEY",
        "model_name": "inertial-embed-alpha",
        "dimensions": 512,
    }
    ef = InertialAIEmbeddingFunction.build_from_config(config)
    assert ef.dimensions == 512
