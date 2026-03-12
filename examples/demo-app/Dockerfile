FROM python:3.11-slim

# Install uv from the official distroless image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Install into the system Python so `streamlit` is available on PATH
ENV UV_SYSTEM_PYTHON=1

WORKDIR /app

# Install dependencies declared in the scripts' inline metadata.
# Copied before the app files so Docker layer-caches this step independently.
COPY ingest.py app.py ./
RUN uv pip install \
    "inertialai-chroma>=0.1.0" \
    "chromadb>=1.0.0" \
    "streamlit>=1.30.0" \
    "scipy>=1.10.0" \
    "plotly>=5.0.0" \
    "pandas>=2.0.0"

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
