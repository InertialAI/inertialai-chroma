# Changelog

All notable changes to `inertialai-chroma` will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/). This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

---

## [0.1.0] - 2026-03-11

Initial release.

### Added

- `InertialAIEmbeddingFunction` — a full implementation of Chroma's eight-method `EmbeddingFunction[Documents]` interface backed by the InertialAI embeddings API.
- **Text-only embeddings** — plain string documents are automatically wrapped as `{"text": <string>}` before being sent to the API.
- **Time-series-only and multi-modal embeddings** — documents passed as JSON-serialised dicts (e.g. `json.dumps({"text": ..., "time_series": ...})`) are decoded and forwarded as-is, enabling InertialAI's multi-modal embedding capability within Chroma's string-based document model.
- **Chroma collection persistence** — `get_config()` and `build_from_config()` allow collections to be saved to disk and reconstructed without storing credentials. The environment variable *name* is serialised, never the key value.
- **Immutability enforcement** — `validate_config_update()` raises `ValueError` if `model_name` or `dimensions` are changed after a collection has been created, preventing silent index invalidation.
- `api_key_env_var` parameter (default: `"INERTIALAI_API_KEY"`) — resolves the API key from the environment at construction time.
- `model_name` parameter (default: `"inertial-embed-alpha"`).
- `dimensions` parameter — optional integer to truncate embedding output size.
- `timeout` parameter (default: `60.0` seconds) — controls the HTTP request timeout and is included in the persisted config.
- `DeprecationWarning` when `api_key` is passed directly, guiding users toward the environment variable approach.
- Embeddings returned as `list[numpy.ndarray]` with `dtype=float32`.
- PEP 561 `py.typed` marker — downstream type checkers recognise that this package ships inline types.

[Unreleased]: https://github.com/InertialAI/inertialai-chroma/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/InertialAI/inertialai-chroma/releases/tag/v0.1.0
