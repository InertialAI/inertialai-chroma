# ECG5000 Multi-Modal Embeddings Demo

This example demonstrates **InertialAI's multi-modal embeddings** — the ability to fuse text and time-series data into a single dense vector. Each document is a JSON-serialized pair of a human-readable ECG class description and 140 raw sensor readings from the [ECG5000 dataset](https://timeseriesclassification.com/description.php?Dataset=ECG5000).

The demo runs a **Chroma** vector database alongside a **Streamlit** web UI, all orchestrated via Docker Compose.

---

## Prerequisites

| Tool                                                       | Purpose                                |
| ---------------------------------------------------------- | -------------------------------------- |
| [Docker](https://docs.docker.com/get-docker/) + Compose v2 | Run Chroma and the demo app            |
| `make`                                                     | Convenience targets                    |
| `curl` + `unzip`                                           | Used by `data.sh` to fetch the dataset |
| `INERTIALAI_API_KEY`                                       | Embed documents via the InertialAI API |

---

## Quick Start

### 1. Download the sample data

```bash
make example-data-fetch
```

This fetches `ECG5000.zip` from the UCR archive and extracts the two ARFF files into `examples/sample-data/ecg5000/`.

### 2. Configure your API key

```bash
make example-init-env
# Edit examples/.env and replace `your_api_key_here` with your real key
```

### 3. Start the services

```bash
make example-up
```

Docker Compose will:

1. Pull and start the Chroma server (port **8000**)
2. Build the demo-app image and start Streamlit (port **8501**)
3. On first load, the app auto-ingests ~200 ECG samples (calls the InertialAI API)

Open **http://localhost:8501** in your browser.

---

## Services

| Service  | URL                   | Description                                    |
| -------- | --------------------- | ---------------------------------------------- |
| Chroma   | http://localhost:8000 | Vector database (persistent via Docker volume) |
| Demo app | http://localhost:8501 | Streamlit search + browse UI                   |

---

## Using the Streamlit UI

### Search tab

Enter a free-text query. The app embeds it via InertialAI, queries Chroma for the nearest neighbours, and returns a ranked table of results with similarity scores and class labels, plus ECG waveform charts for the top hits.

The queries below are chosen to highlight specific capabilities of multi-modal embeddings.

#### Normal vs. abnormal discrimination

| Query           | What to expect                                                                                                           |
| --------------- | ------------------------------------------------------------------------------------------------------------------------ |
| `healthy heart` | Results dominated by class 1 (Normal sinus rhythm). Abnormal classes ranking highly would indicate weak signal encoding. |
| `abnormal ECG`  | The inverse — class 1 should rank at the bottom, abnormal classes at the top.                                            |

#### Arrhythmia specificity

| Query                       | What to expect                                                                                                                                                     |
| --------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `premature heartbeat`       | R-on-T PVCs (class 2) should rank first; other abnormal classes may follow, but normal rhythms should rank last.                                                   |
| `artificially paced rhythm` | A clean test — paced beats (class 3) have a visually distinct flat, square-wave morphology and a dedicated class label, so they should cluster tightly at the top. |

#### ST-segment differentiation

Classes 4 and 5 are easy to confuse by text alone — this is where the time-series component earns its place.

| Query               | What to expect                                                    |
| ------------------- | ----------------------------------------------------------------- |
| `ischemia`          | ST depression (class 4) should rank above ST elevation (class 5). |
| `myocardial injury` | The reverse — ST elevation (class 5) should lead.                 |

#### Cross-modal retrieval

The most compelling showcase: queries that use language not present in any stored document description, relying entirely on the fused text + signal embedding to find the right results.

| Query                        | What to expect                                                                                                    |
| ---------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| `cardiac emergency`          | Abnormal classes should surface, with ST elevation and PVCs ranking highest.                                      |
| `heart attack warning signs` | Tests whether the model maps lay medical terminology to the correct ECG morphologies.                             |
| `elevated ST segment`        | Should surface ST elevation samples (class 5) even though the exact phrase doesn't appear in stored descriptions. |

### Browse tab

Paginated view of all ingested documents — useful for exploring the dataset and verifying ingest quality.

### Sidebar

- **Collection stats** — total document count and per-class breakdown
- **Re-ingest** button — deletes and re-embeds all documents (useful after changing sample size)

---

## Running Ingestion Standalone (outside Docker)

`ingest.py` declares its dependencies as [PEP 723 inline script metadata](https://peps.python.org/pep-0723/), so `uv` can install and run it in one step — no manual `pip install` needed:

```bash
# Run ingestion (Chroma must be running on localhost:8000)
uv run examples/demo-app/ingest.py --host localhost --port 8000

# Options
uv run examples/demo-app/ingest.py --help
```

### Ingestion CLI options

| Flag            | Default     | Description                               |
| --------------- | ----------- | ----------------------------------------- |
| `--host`        | `localhost` | Chroma server host                        |
| `--port`        | `8000`      | Chroma server port                        |
| `--sample-size` | `200`       | Total documents to ingest (~40 per class) |
| `--force`       | off         | Delete existing collection and re-ingest  |

---

## Makefile Targets

| Target                          | Description                                     |
| ------------------------------- | ----------------------------------------------- |
| `make example-data-fetch`       | Download ECG5000 sample data                    |
| `make example-data-fetch-force` | Force re-download sample data                   |
| `make example-data-clean`       | Remove downloaded sample data                   |
| `make example-init-env`         | Copy `.env.example` → `.env` (skips if exists)  |
| `make example-up`               | Start Chroma + demo app (detached)              |
| `make example-down`             | Stop all services                               |
| `make example-logs`             | Stream service logs                             |
| `make example-reset`            | Stop services and wipe Chroma volumes           |
| `make example-clean`            | Full cleanup (services + volumes + sample data) |

---

## Troubleshooting

**App shows "Sample data not found"**
→ Run `make example-data-fetch` on your host, then `make example-reset && make example-up`.

**App shows "Ingestion failed: connection refused"**
→ The Chroma container may not be healthy yet. Check `make example-logs` and wait for `chroma` to pass its healthcheck.

**"INERTIALAI_API_KEY not set" error**
→ Make sure `examples/.env` exists and contains your key. The `.env` file is mounted into the demo-app container via `env_file` in `docker-compose.yml`.

**Want to re-embed with a different sample size?**
→ Use the Re-ingest button in the sidebar (uses the default 200-doc sample), or run `ingest.py --force --sample-size N` standalone.
