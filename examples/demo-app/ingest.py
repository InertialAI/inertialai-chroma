# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "inertialai-chroma>=0.1.0",
#   "chromadb>=1.0.0",
#   "scipy>=1.10.0",
# ]
# ///
"""
ingest.py — Ingest ECG5000 multi-modal data into a Chroma collection.

Parses ECG5000_TRAIN.arff and ECG5000_TEST.arff, builds multi-modal documents
combining text class descriptions with raw time-series readings, and populates
a Chroma collection named 'ecg5000' via InertialAIEmbeddingFunction.

Usage (CLI):
    uv run ingest.py [--host HOST] [--port PORT] [--sample-size N] [--force]

Importable:
    from ingest import run_ingestion
    run_ingestion(host="localhost", port=8000)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import NamedTuple

import chromadb
from scipy.io.arff import loadarff

from inertialai_chroma import InertialAIEmbeddingFunction

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

COLLECTION_NAME = "ecg5000"
DATA_DIR = Path("/app/sample-data/ecg5000")
ARFF_FILES = {
    "train": DATA_DIR / "ECG5000_TRAIN.arff",
    "test": DATA_DIR / "ECG5000_TEST.arff",
}
DEFAULT_SAMPLE_SIZE = 200
SAMPLES_PER_CLASS = DEFAULT_SAMPLE_SIZE // 5  # 5 ECG classes → 40 each

# Maps integer class label (stored as float in ARFF) to human-readable description
CLASS_DESCRIPTIONS: dict[int, str] = {
    1: "Normal sinus rhythm — regular heartbeat pattern",
    2: "R-on-T premature ventricular contraction — early heartbeat on T-wave",
    3: "Paced beat — artificially stimulated heartbeat pattern",
    4: "ST depression — ischemic episode with depressed ST segment",
    5: "ST elevation — potential myocardial injury with elevated ST segment",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


class ECGSample(NamedTuple):
    sample_id: str
    class_label: int
    time_series: list[list[float]]  # list of channels; ECG5000 is single-channel
    source: str


# ---------------------------------------------------------------------------
# ARFF parsing
# ---------------------------------------------------------------------------


def _load_arff(path: Path, source_tag: str) -> list[ECGSample]:
    """
    Parse a single ARFF file into a list of ECGSample objects.
    """
    data, _ = loadarff(path)
    samples: list[ECGSample] = []

    for idx, row in enumerate(data):
        values = list(row)
        # Last column is the class label (stored as float)
        class_label = int(float(values[-1]))
        # ECG5000 is single-channel; wrap in outer list to match the API's
        # list[list[float]] (list of channels) schema.
        time_series = [[float(v) for v in values[:-1]]]
        sample_id = f"{source_tag}_{idx}"
        samples.append(ECGSample(sample_id, class_label, time_series, source_tag))

    return samples


def load_ecg_samples() -> list[ECGSample]:
    """
    Load and combine train + test ARFF files.
    """
    all_samples: list[ECGSample] = []
    for source, path in ARFF_FILES.items():
        if not path.exists():
            raise FileNotFoundError(
                f"ARFF file not found: {path}\n"
                "Run 'make example-data-fetch' to download the ECG5000 dataset."
            )
        log.info("Loading %s from %s", source, path)
        all_samples.extend(_load_arff(path, source))
    return all_samples


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def stratified_sample(samples: list[ECGSample], per_class: int) -> list[ECGSample]:
    """
    Return up to `per_class` samples for each class label, preserving order.
    """
    counts: dict[int, int] = {}
    selected: list[ECGSample] = []

    for sample in samples:
        label = sample.class_label
        if counts.get(label, 0) < per_class:
            selected.append(sample)
            counts[label] = counts.get(label, 0) + 1

    return selected


# ---------------------------------------------------------------------------
# Document construction
# ---------------------------------------------------------------------------


def build_document(sample: ECGSample) -> str:
    """
    Serialise a multi-modal document as a JSON string for InertialAIEmbeddingFunction.
    """
    description = CLASS_DESCRIPTIONS.get(sample.class_label, f"Unknown class {sample.class_label}")
    payload = {
        "text": description,
        "time_series": sample.time_series,
    }
    return json.dumps(payload)


# ---------------------------------------------------------------------------
# Chroma helpers
# ---------------------------------------------------------------------------


def get_chroma_client(host: str, port: int) -> chromadb.HttpClient:
    return chromadb.HttpClient(host=host, port=port)


def get_or_create_collection(
    client: chromadb.HttpClient,
) -> chromadb.Collection:
    embedding_fn = InertialAIEmbeddingFunction()
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
        metadata={"description": "ECG5000 multi-modal embeddings (text + time series)"},
    )


# ---------------------------------------------------------------------------
# Core ingestion logic
# ---------------------------------------------------------------------------


def run_ingestion(
    host: str = "localhost",
    port: int = 8000,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    force: bool = False,
) -> int:
    """
    Ingest ECG5000 data into Chroma.

    Returns the total number of documents in the collection after ingestion. Skips
    ingestion if the collection is already populated (unless force=True).
    """
    client = get_chroma_client(host, port)
    collection = get_or_create_collection(client)
    existing_count = collection.count()

    if existing_count > 0 and not force:
        log.info(
            "Collection '%s' already contains %d documents — skipping ingestion. "
            "Pass --force to re-ingest.",
            COLLECTION_NAME,
            existing_count,
        )
        return existing_count

    if force and existing_count > 0:
        log.info("Force flag set — deleting existing collection before re-ingesting.")
        client.delete_collection(COLLECTION_NAME)
        collection = get_or_create_collection(client)

    per_class = sample_size // len(CLASS_DESCRIPTIONS)
    log.info(
        "Loading ECG5000 samples (target: %d total, %d per class) ...",
        sample_size,
        per_class,
    )

    all_samples = load_ecg_samples()
    selected = stratified_sample(all_samples, per_class=per_class)
    log.info("Selected %d samples across %d classes.", len(selected), len(CLASS_DESCRIPTIONS))

    ids = [s.sample_id for s in selected]
    documents = [build_document(s) for s in selected]
    metadatas = [{"class_label": s.class_label, "source": s.source} for s in selected]

    log.info("Embedding and ingesting %d documents into Chroma ...", len(selected))
    # Chroma recommends batches ≤ 5 000; our sample is small, so a single call is fine
    collection.add(ids=ids, documents=documents, metadatas=metadatas)

    final_count = collection.count()
    log.info(
        "Ingestion complete. Collection '%s' now has %d documents.", COLLECTION_NAME, final_count
    )
    return final_count


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ingest ECG5000 multi-modal data into a Chroma collection.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--host", default="localhost", help="Chroma server host")
    parser.add_argument("--port", type=int, default=8000, help="Chroma server port")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help="Total number of documents to ingest (distributed across all classes)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete the existing collection and re-ingest from scratch",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    try:
        count = run_ingestion(
            host=args.host,
            port=args.port,
            sample_size=args.sample_size,
            force=args.force,
        )
        print(f"Collection '{COLLECTION_NAME}' contains {count} documents.")
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"Ingestion failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
