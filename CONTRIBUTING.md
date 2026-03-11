# Contributing to inertialai-chroma

## Prerequisites

- Python 3.9+
- [uv](https://docs.astral.sh/uv/) — used for dependency management and running all tooling

## Setup

```bash
git clone https://github.com/InertialAI/inertialai-chroma.git
cd inertialai-chroma
make install
```

`make install` installs all dependencies (including dev dependencies) and sets up the pre-commit hooks.

## Running Tests

```bash
make test        # unit tests with coverage
make test-dev    # unit tests with verbose output
```

Unit tests use `respx` to mock the HTTP layer — no API key is required.

### Integration Tests

Integration tests hit the live InertialAI API and require a valid API key:

```bash
export INERTIALAI_API_KEY=your_api_key_here
make test-integration
```

## Code Quality

```bash
make lint-format   # lint, auto-fix, and format
make type-check    # mypy in strict mode
make check         # lint + type-check + test
```

Pre-commit hooks run `ruff` and `mypy` automatically on commit, and the full test suite runs on push.

## Submitting a Pull Request

1. Fork the repository and create a branch from `main`.
2. Make your changes and ensure `make check` passes.
3. Open a pull request against `main`.
