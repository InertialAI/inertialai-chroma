# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "inertialai-chroma>=0.1.0",
#   "chromadb>=1.0.0",
#   "streamlit>=1.30.0",
#   "scipy>=1.10.0",
#   "plotly>=5.0.0",
#   "pandas>=2.0.0",
# ]
# ///
"""
app.py — Streamlit web application for the ECG5000 multi-modal embeddings demo.

Features:
  - Auto-ingests ECG5000 data on cold start (collection empty or missing).
  - Search tab: semantic text query → ranked results + ECG waveform charts.
  - Browse tab: paginated table of all ingested documents with metadata.

Environment variables:
  CHROMA_HOST  (default: localhost)
  CHROMA_PORT  (default: 8000)
"""

from __future__ import annotations

import json
import os
from typing import Any

import chromadb
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from ingest import (
    CLASS_DESCRIPTIONS,
    COLLECTION_NAME,
    run_ingestion,
)

from inertialai_chroma import InertialAIEmbeddingFunction

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CHROMA_HOST = os.environ.get("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.environ.get("CHROMA_PORT", "8000"))
SEARCH_N_RESULTS = 5
BROWSE_PAGE_SIZE = 20

# ---------------------------------------------------------------------------
# Cached resources (survive Streamlit reruns)
# ---------------------------------------------------------------------------


@st.cache_resource
def get_chroma_client() -> chromadb.HttpClient:
    return chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)


@st.cache_resource
def get_embedding_function() -> InertialAIEmbeddingFunction:
    return InertialAIEmbeddingFunction()


def get_collection() -> chromadb.Collection:
    """
    Fetch the collection each time (not cached — count changes after ingest).
    """
    client = get_chroma_client()
    embedding_fn = get_embedding_function()
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
        metadata={"description": "ECG5000 multi-modal embeddings (text + time series)"},
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def parse_document(doc: str) -> dict[str, Any]:
    """
    Parse a JSON document string back into a dict.
    """
    try:
        return json.loads(doc)
    except (json.JSONDecodeError, TypeError):
        return {"text": doc, "time_series": []}


def class_label_to_name(label: int) -> str:
    return CLASS_DESCRIPTIONS.get(label, f"Unknown ({label})")


def render_ecg_chart(time_series: list[float], title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            y=time_series,
            mode="lines",
            line={"color": "#1f77b4", "width": 1.5},
            name="ECG signal",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Sample index",
        yaxis_title="Amplitude",
        height=200,
        margin={"l": 40, "r": 20, "t": 40, "b": 30},
        showlegend=False,
    )
    return fig


# ---------------------------------------------------------------------------
# Auto-ingest on cold start
# ---------------------------------------------------------------------------


def maybe_auto_ingest() -> None:
    """
    Run ingestion if the collection is empty; show a spinner while working.
    """
    collection = get_collection()
    if collection.count() > 0:
        return

    st.info("Collection is empty — ingesting ECG5000 data on first run. This may take a minute.")
    with st.spinner("Embedding and ingesting ECG5000 samples …"):
        try:
            count = run_ingestion(host=CHROMA_HOST, port=CHROMA_PORT)
            st.success(f"Ingested {count} documents successfully.")
            st.rerun()
        except FileNotFoundError as exc:
            st.error(
                f"Sample data not found: {exc}\n\n"
                "Run `make example-data-fetch` on your host machine, then restart the app."
            )
            st.stop()
        except Exception as exc:
            st.error(f"Ingestion failed: {exc}")
            st.stop()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------


def render_sidebar() -> dict[str, Any]:
    """
    Render the sidebar and return user settings.
    """
    st.sidebar.title("ECG5000 Demo")
    st.sidebar.caption("InertialAI multi-modal embeddings")
    st.sidebar.divider()

    # Collection stats
    collection = get_collection()
    count = collection.count()
    st.sidebar.metric("Documents in collection", count)

    if count > 0:
        results = collection.get(include=["metadatas"])
        metadatas = results.get("metadatas") or []
        class_counts: dict[str, int] = {}
        for meta in metadatas:
            label = int(meta.get("class_label", 0))
            name = CLASS_DESCRIPTIONS.get(label, f"Class {label}").split(" — ")[0]
            class_counts[name] = class_counts.get(name, 0) + 1

        if class_counts:
            st.sidebar.caption("Class breakdown")
            for cls_name, cls_count in sorted(class_counts.items()):
                st.sidebar.text(f"  {cls_name}: {cls_count}")

    st.sidebar.divider()

    # Re-ingest button
    if st.sidebar.button("Re-ingest data", help="Delete and re-embed all documents"):
        with st.spinner("Re-ingesting …"):
            try:
                new_count = run_ingestion(host=CHROMA_HOST, port=CHROMA_PORT, force=True)
                st.sidebar.success(f"Re-ingested {new_count} documents.")
                st.rerun()
            except Exception as exc:
                st.sidebar.error(f"Re-ingest failed: {exc}")

    # Settings
    st.sidebar.divider()
    n_results = st.sidebar.slider(
        "Search results to show", min_value=1, max_value=20, value=SEARCH_N_RESULTS
    )
    return {"n_results": n_results}


# ---------------------------------------------------------------------------
# Search tab
# ---------------------------------------------------------------------------


def render_search_tab(settings: dict[str, Any]) -> None:
    st.header("Semantic Search")
    st.caption(
        "Query by text description. Results are ranked by vector similarity "
        "using multi-modal ECG embeddings."
    )

    query = st.text_input(
        "Search query",
        placeholder='e.g. "irregular heartbeat" or "normal sinus rhythm"',
    )

    if not query:
        return

    collection = get_collection()
    if collection.count() == 0:
        st.warning("Collection is empty. Check the sidebar to re-ingest.")
        return

    with st.spinner("Searching …"):
        results = collection.query(
            query_texts=[query],
            n_results=min(settings["n_results"], collection.count()),
            include=["documents", "metadatas", "distances"],
        )

    documents = (results.get("documents") or [[]])[0]
    metadatas = (results.get("metadatas") or [[]])[0]
    distances = (results.get("distances") or [[]])[0]

    if not documents:
        st.info("No results found.")
        return

    # Results table
    rows = []
    for rank, (doc, meta, dist) in enumerate(
        zip(documents, metadatas, distances, strict=True), start=1
    ):
        parsed = parse_document(doc)
        rows.append(
            {
                "Rank": rank,
                "Similarity": round(1 - dist, 4),
                "Class": class_label_to_name(int(meta.get("class_label", 0))),
                "Source": meta.get("source", ""),
            }
        )

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Waveform charts for top results
    st.subheader("ECG Waveforms — Top Results")
    for rank, (doc, meta) in enumerate(zip(documents, metadatas, strict=True), start=1):
        parsed = parse_document(doc)
        channels = parsed.get("time_series", [])
        # time_series is list[list[float]] (channels); render the first channel
        ts = channels[0] if channels else []
        if not ts:
            continue
        label = int(meta.get("class_label", 0))
        title = f"#{rank} — {class_label_to_name(label)}"
        st.plotly_chart(
            render_ecg_chart(ts, title),
            use_container_width=True,
            key=f"waveform_{rank}",
        )


# ---------------------------------------------------------------------------
# Browse tab
# ---------------------------------------------------------------------------


def render_browse_tab() -> None:
    st.header("Browse All Documents")

    collection = get_collection()
    total = collection.count()

    if total == 0:
        st.info("No documents in the collection yet.")
        return

    total_pages = max(1, (total + BROWSE_PAGE_SIZE - 1) // BROWSE_PAGE_SIZE)
    page = st.number_input("Page", min_value=1, max_value=total_pages, value=1, step=1)
    offset = (page - 1) * BROWSE_PAGE_SIZE

    results = collection.get(
        include=["documents", "metadatas"],
        limit=BROWSE_PAGE_SIZE,
        offset=offset,
    )

    ids = results.get("ids") or []
    documents = results.get("documents") or []
    metadatas = results.get("metadatas") or []

    rows = []
    for doc_id, doc, meta in zip(ids, documents, metadatas, strict=True):
        parsed = parse_document(doc)
        rows.append(
            {
                "ID": doc_id,
                "Class": class_label_to_name(int(meta.get("class_label", 0))),
                "Source": meta.get("source", ""),
                "Text": parsed.get("text", ""),
                "TS length": len(parsed.get("time_series", [])),
            }
        )

    st.caption(f"Showing {offset + 1}–{min(offset + BROWSE_PAGE_SIZE, total)} of {total}")
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(
        page_title="ECG5000 — InertialAI Multi-Modal Embeddings",
        page_icon="💓",
        layout="wide",
    )

    maybe_auto_ingest()

    settings = render_sidebar()

    search_tab, browse_tab = st.tabs(["Search", "Browse"])
    with search_tab:
        render_search_tab(settings)
    with browse_tab:
        render_browse_tab()


if __name__ == "__main__":
    main()
