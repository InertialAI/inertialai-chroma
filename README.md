# inertialai-chroma

[![PyPI](https://img.shields.io/pypi/v/inertialai-chroma)](https://pypi.org/project/inertialai-chroma/)
[![Python](https://img.shields.io/pypi/pyversions/inertialai-chroma)](https://pypi.org/project/inertialai-chroma/)
[![License](https://img.shields.io/pypi/l/inertialai-chroma)](LICENSE)

[InertialAI](https://www.inertialai.com) embeddings integration for [Chroma](https://www.trychroma.com/) — supports text-only, time-series-only, and **multi-modal** (text + time-series) semantic search via the [`inertial-embed-alpha`](https://docs.inertialai.com/docs/using-the-embeddings-endpoint) model.

---

## Overview

InertialAI's `inertial-embed-alpha` model produces dense vector embeddings from three input types:

| Input type           | Example                                                                                          |
| -------------------- | ------------------------------------------------------------------------------------------------ |
| **Text only**        | `"temperature spike at noon"`                                                                    |
| **Time-series only** | `[[72, 74, 73, 75, 71]]` (e.g. heart rate readings)                                              |
| **Multi-modal**      | A time-series signal paired with its natural language description, combined into a single vector |

Multi-modal embeddings are InertialAI's core differentiator — a single vector that captures both the numerical signal and its semantic context simultaneously, enabling richer similarity search across domains like industrial IoT, healthcare, and financial markets.

`inertialai-chroma` implements Chroma's full eight-method `EmbeddingFunction` interface, including `get_config()` and `build_from_config()`, so that collections can be persisted to disk and reconstructed without storing credentials.

---

## Requirements

- Python 3.11 or later
- An InertialAI API key — [sign up at app.inertialai.com](https://app.inertialai.com)
- A running Chroma instance — see [Chroma Docker deployment](https://docs.trychroma.com/guides/deploy/docker)

---

## Installation

```bash
pip install inertialai-chroma
# or
uv add inertialai-chroma
```

---

## Quickstart

Set your API key as an environment variable:

```bash
export INERTIALAI_API_KEY="your-api-key"
```

Then create a collection and start embedding:

```python
import chromadb
from inertialai_chroma import InertialAIEmbeddingFunction

client = chromadb.HttpClient(host="localhost", port=8000)
ef = InertialAIEmbeddingFunction()  # reads INERTIALAI_API_KEY from env

collection = client.create_collection("sensors", embedding_function=ef)

collection.add(
    documents=[
        "temperature spike at noon",
        "stable overnight readings",
        "pressure anomaly detected",
    ],
    ids=["doc-1", "doc-2", "doc-3"],
)

results = collection.query(query_texts=["anomalous thermal event"], n_results=2)
print(results["documents"])
```

---

## Multi-modal Embeddings

Since Chroma documents are always strings, multi-modal inputs are passed as JSON-serialised dicts containing a `text` field, a `time_series` field, or both. Time-series data is formatted as a list of channels, where each channel is a list of numerical readings.

```python
import json
import chromadb
from inertialai_chroma import InertialAIEmbeddingFunction

client = chromadb.HttpClient(host="localhost", port=8000)
ef = InertialAIEmbeddingFunction()
collection = client.create_collection("energy", embedding_function=ef)

# Each document pairs a text description with its raw time-series readings
collection.add(
    documents=[
        json.dumps({
            "text": "Energy price spike Q4 2022",
            "time_series": [[1.2, 1.5, 1.8, 2.1, 2.4]],
        }),
        json.dumps({
            "text": "Stable energy prices Q1 2023",
            "time_series": [[0.9, 0.9, 0.91, 0.88, 0.9]],
        }),
    ],
    ids=["doc-1", "doc-2"],
)

# Query with a multi-modal input — or just plain text
results = collection.query(
    query_texts=[
        json.dumps({"text": "abnormal energy reading", "time_series": [[1.3, 1.6, 2.0]]})
    ],
    n_results=1,
)
print(results["documents"])
```

---

## Configuration

```python
InertialAIEmbeddingFunction(
    api_key_env_var="INERTIALAI_API_KEY",  # default
    model_name="inertial-embed-alpha",     # default
    dimensions=None,                        # default — use full embedding size
    timeout=60.0,                           # default — seconds
)
```

| Parameter         | Type          | Default                  | Description                                                                                    |
| ----------------- | ------------- | ------------------------ | ---------------------------------------------------------------------------------------------- |
| `api_key_env_var` | `str`         | `"INERTIALAI_API_KEY"`   | Name of the environment variable holding the API key                                           |
| `model_name`      | `str`         | `"inertial-embed-alpha"` | InertialAI model to use for embedding                                                          |
| `dimensions`      | `int \| None` | `None`                   | Truncate embedding output to this many dimensions                                              |
| `timeout`         | `float`       | `60.0`                   | HTTP request timeout in seconds                                                                |
| `api_key`         | `str \| None` | `None`                   | _(Deprecated)_ Pass the key value directly — not persisted by Chroma; prefer `api_key_env_var` |

> **Note:** `model_name` and `dimensions` are immutable after a collection is created — changing either would invalidate the existing vector index.

---

## Collection Persistence

When Chroma persists a collection to disk, it serialises the embedding function via `get_config()`. `InertialAIEmbeddingFunction` stores the **environment variable name**, never the key value itself, so serialised collections are safe to commit to version control. At load time, `build_from_config()` resolves the key from the environment automatically — no credentials need to be passed explicitly.

---

## Links

- [InertialAI embeddings API guide](https://docs.inertialai.com/docs/using-the-embeddings-endpoint)
- [InertialAI website](https://www.inertialai.com)
- [Chroma EmbeddingFunction docs](https://docs.trychroma.com/docs/embeddings/embedding-functions)
- [Chroma Docker deployment](https://docs.trychroma.com/guides/deploy/docker)
- [GitHub repository](https://github.com/InertialAI/inertialai-chroma)
