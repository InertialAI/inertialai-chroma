"""
Integration tests for InertialAIEmbeddingFunction.

These tests make real HTTP calls to the InertialAI API and require a live Chroma instance.
They are automatically skipped when INERTIALAI_API_KEY is not set in the environment.
"""

from __future__ import annotations

import json
import os
from collections.abc import Generator

import numpy as np
import pytest

from inertialai_chroma import InertialAIEmbeddingFunction


@pytest.fixture(autouse=True)
def require_api_key() -> None:
    if not os.environ.get("INERTIALAI_API_KEY"):
        pytest.skip("INERTIALAI_API_KEY not set")


@pytest.fixture()
def ef() -> InertialAIEmbeddingFunction:
    return InertialAIEmbeddingFunction()


class TestLiveAPI:
    def test_text_only_embedding(self, ef: InertialAIEmbeddingFunction) -> None:
        result = ef(["energy price spike in Q4 2022"])
        assert len(result) == 1
        assert isinstance(result[0], np.ndarray)
        assert result[0].dtype == np.float32
        assert result[0].ndim == 1
        assert len(result[0]) > 0

    def test_batch_text_embeddings(self, ef: InertialAIEmbeddingFunction) -> None:
        docs = ["stable overnight readings", "temperature spike at noon"]
        result = ef(docs)
        assert len(result) == 2
        for emb in result:
            assert isinstance(emb, np.ndarray)
            assert emb.dtype == np.float32

    def test_multimodal_embedding(self, ef: InertialAIEmbeddingFunction) -> None:
        doc = json.dumps(
            {"text": "Energy price spike Q4 2022", "time_series": [[1.2, 1.5, 1.8, 2.1]]}
        )
        result = ef([doc])
        assert len(result) == 1
        assert isinstance(result[0], np.ndarray)
        assert result[0].dtype == np.float32

    def test_empty_input(self, ef: InertialAIEmbeddingFunction) -> None:
        with pytest.raises(ValueError):
            ef([])

    def test_dimensions_truncation(self) -> None:
        ef_full = InertialAIEmbeddingFunction()
        ef_trunc = InertialAIEmbeddingFunction(dimensions=64)

        full = ef_full(["test"])
        trunc = ef_trunc(["test"])

        assert len(full[0]) >= len(trunc[0])
        assert len(trunc[0]) == 64


class TestLiveChroma:
    """
    End-to-end tests against a live Chroma instance.

    Requires CHROMA_HOST and CHROMA_PORT in the environment (or defaults to
    localhost:8000).  Skipped automatically if the Chroma server is unreachable.
    """

    @pytest.fixture()
    def chroma_client(self) -> Generator[chromadb.HttpClient, None, None]:  # type: ignore[name-defined]  # noqa: F821
        try:
            import chromadb
        except ImportError:
            pytest.skip("chromadb not installed")

        host = os.environ.get("CHROMA_HOST", "localhost")
        port = int(os.environ.get("CHROMA_PORT", "8000"))
        try:
            client = chromadb.HttpClient(host=host, port=port)
            client.heartbeat()
        except Exception:
            pytest.skip(f"Chroma not reachable at {host}:{port}")
        yield client
        client.close()  # type: ignore[attr-defined]
        client.clear_system_cache()

    def test_collection_add_and_query(
        self,
        ef: InertialAIEmbeddingFunction,
        chroma_client: chromadb.HttpClient,  # type: ignore[name-defined]  # noqa: F821
    ) -> None:
        import uuid

        collection_name = f"test-{uuid.uuid4().hex[:8]}"
        try:
            collection = chroma_client.create_collection(collection_name, embedding_function=ef)
            collection.add(
                documents=["energy price spike Q4 2022", "stable overnight readings"],
                ids=["doc-1", "doc-2"],
            )
            results = collection.query(query_texts=["anomalous energy event"], n_results=1)
            assert len(results["ids"][0]) == 1
        finally:
            chroma_client.delete_collection(collection_name)
