from __future__ import annotations

import json
import os
import warnings
from typing import Any

import httpx
import numpy as np
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings, Space

_API_URL = "https://inertialai.com/api/v1/embeddings"


class InertialAIEmbeddingFunction(EmbeddingFunction[Documents]):
    """
    Chroma EmbeddingFunction backed by the InertialAI embeddings API.

    Supports text-only, time-series-only, and multi-modal inputs. When a
    document is a plain string it is wrapped as ``{"text": <string>}``. When a
    document is a JSON-serialised dict (produced by
    ``json.dumps({"text": ..., "time_series": ...})``) it is decoded and sent
    as-is, enabling InertialAI's multi-modal embedding capability.
    """

    def __init__(
        self,
        api_key: str | None = None,
        api_key_env_var: str = "INERTIALAI_API_KEY",
        model_name: str = "inertial-embed-alpha",
        dimensions: int | None = None,
        timeout: float = 60.0,
    ) -> None:
        """
        Initialize the InertialAIEmbeddingFunction.

        Args:
            api_key (str, optional): API key value. Deprecated — passing the key
                directly will not be persisted by Chroma's collection serialisation.
                Use ``api_key_env_var`` instead.
            api_key_env_var (str, optional): Name of the environment variable that
                holds the InertialAI API key. Defaults to ``"INERTIALAI_API_KEY"``.
                This name (not the key value) is stored in ``get_config()`` so that
                persisted collections never contain credentials.
            model_name (str, optional): InertialAI model to use for embedding.
                Defaults to ``"inertial-embed-alpha"``.
            dimensions (int, optional): Truncate embedding output to this many
                dimensions. When ``None`` (default) the full embedding size is
                returned. Cannot be changed after a collection has been created.
            timeout (float, optional): HTTP request timeout in seconds for calls to
                the InertialAI API. Defaults to ``60.0``.

        Raises:
            ValueError: If the resolved API key is empty (i.e. the environment
                variable is not set and no ``api_key`` was provided).
            DeprecationWarning: If ``api_key`` is passed directly.
        """
        if api_key is not None:
            warnings.warn(
                "Direct api_key configuration will not be persisted. "
                "Please use environment variables via api_key_env_var for persistent storage.",
                DeprecationWarning,
                stacklevel=2,
            )

        self.api_key_env_var = api_key_env_var

        self.api_key = api_key or os.getenv(api_key_env_var)

        if not self.api_key:
            raise ValueError(f"The {api_key_env_var} environment variable is not set.")

        self.model_name = model_name
        self.dimensions = dimensions
        self.timeout = timeout

    def __call__(self, input: Documents) -> Embeddings:
        """
        Generate embeddings for the given documents.

        Each document is transformed into an API input object before the request
        is sent:

        - **Plain string** → ``{"text": string}``
        - **JSON-serialised dict** → decoded and forwarded as-is, enabling
          multi-modal inputs such as ``{"text": ..., "time_series": ...}``.

        Detection is performed by attempting ``json.loads()`` on every item. If
        parsing succeeds and the result is a ``dict``, it is used directly;
        otherwise the item is treated as plain text.

        Args:
            input: A list of document strings to embed.

        Returns:
            A list of ``numpy.ndarray`` vectors with ``dtype=float32``, one per
            input document.

        Raises:
            httpx.HTTPStatusError: If the InertialAI API returns a non-2xx
                response. Errors propagate directly without custom wrapping.
        """
        if not input:
            return []

        api_inputs: list[dict[str, Any]] = []

        for item in input:
            try:
                parsed = json.loads(item)

                if isinstance(parsed, dict):
                    api_inputs.append(parsed)
                else:
                    api_inputs.append({"text": item})

            except json.JSONDecodeError:
                api_inputs.append({"text": item})

        body: dict[str, Any] = {
            "model": self.model_name,
            "input": api_inputs,
        }
        if self.dimensions is not None:
            body["dimensions"] = self.dimensions

        response = httpx.post(
            _API_URL,
            headers={"Authorization": f"Bearer {self.api_key}"},
            json=body,
            timeout=self.timeout,
        )

        response.raise_for_status()

        return [np.array(item["embedding"], dtype=np.float32) for item in response.json()["data"]]

    @staticmethod
    def name() -> str:
        """
        Return the name used to identify this embedding function in Chroma's registry.

        Returns:
            The string ``"inertialai"``.
        """
        return "inertialai"

    def default_space(self) -> Space:
        """
        Return the default distance metric for collections using this embedding function.

        Returns:
            ``"cosine"``
        """
        return "cosine"

    def supported_spaces(self) -> list[Space]:
        """
        Return the distance metrics supported by this embedding function.

        Returns:
            A list containing ``"cosine"``, ``"l2"``, and ``"ip"``.
        """
        return ["cosine", "l2", "ip"]

    @staticmethod
    def build_from_config(config: dict[str, Any]) -> InertialAIEmbeddingFunction:
        """
        Reconstruct an InertialAIEmbeddingFunction from a persisted Chroma config.

        Called by Chroma when reopening a collection that was created with this
        embedding function. The API key is resolved from the environment at
        reconstruction time — it is never stored in the config itself.

        Args:
            config: A configuration dict as produced by ``get_config()``.

        Returns:
            A fully initialised ``InertialAIEmbeddingFunction``.
        """
        return InertialAIEmbeddingFunction(
            api_key_env_var=config["api_key_env_var"],
            model_name=config["model_name"],
            dimensions=config.get("dimensions"),
            timeout=config.get("timeout", 60.0),
        )

    def get_config(self) -> dict[str, Any]:
        """
        Serialise the embedding function configuration for Chroma persistence.

        The environment variable *name* is stored rather than the key value,
        so the returned dict is safe to write to disk or commit to version control.

        Returns:
            A dict containing ``api_key_env_var``, ``model_name``, ``dimensions``,
            and ``timeout``.
        """
        return {
            "api_key_env_var": self.api_key_env_var,
            "model_name": self.model_name,
            "dimensions": self.dimensions,
            "timeout": self.timeout,
        }

    def validate_config_update(
        self, old_config: dict[str, Any], new_config: dict[str, Any]
    ) -> None:
        """
        Validate a proposed configuration update against the existing collection config.

        ``model_name`` and ``dimensions`` are immutable after a collection has been
        created — changing either would invalidate the existing vector index.

        Args:
            old_config: The configuration that was used when the collection was created.
            new_config: The proposed updated configuration.

        Raises:
            ValueError: If ``model_name`` or ``dimensions`` differ between
                ``old_config`` and ``new_config``.
        """
        for field in ("model_name", "dimensions"):
            if field in new_config and new_config[field] != old_config.get(field):
                raise ValueError(
                    f"'{field}' cannot be changed after the collection has been created."
                )

    @staticmethod
    def validate_config(config: dict[str, Any]) -> None:  # noqa: ARG004
        """
        Validate a configuration dict for this embedding function.

        Args:
            config: Configuration dict to validate.
        """
        pass
