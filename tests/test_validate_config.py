"""
Unit tests for InertialAIEmbeddingFunction - specifically the
validate_config and validate_config_update methods, ensuring
that they correctly enforce immutability of certain configuration
parameters and that validate_config is a no-op.
"""

from __future__ import annotations

import pytest

from inertialai_chroma import InertialAIEmbeddingFunction

pytestmark = pytest.mark.usefixtures("api_key")


def test_raises_on_model_name_change() -> None:
    ef = InertialAIEmbeddingFunction()
    old = {"model_name": "inertial-embed-alpha", "dimensions": None}
    new = {"model_name": "other-model"}
    with pytest.raises(ValueError, match="model_name"):
        ef.validate_config_update(old, new)


def test_raises_on_dimensions_change() -> None:
    ef = InertialAIEmbeddingFunction(dimensions=256)
    old = {"model_name": "inertial-embed-alpha", "dimensions": 256}
    new = {"dimensions": 512}
    with pytest.raises(ValueError, match="dimensions"):
        ef.validate_config_update(old, new)


def test_no_raise_when_unchanged() -> None:
    ef = InertialAIEmbeddingFunction()
    old = {"model_name": "inertial-embed-alpha", "dimensions": None}
    new = {"model_name": "inertial-embed-alpha"}
    ef.validate_config_update(old, new)  # should not raise


def test_validate_config_is_noop() -> None:
    InertialAIEmbeddingFunction.validate_config({})  # should not raise
    InertialAIEmbeddingFunction.validate_config({"model_name": "anything"})
