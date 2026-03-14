# EPBA — Project Setup Guide

A comprehensive guide for setting up the **Electronic Patient Bedside Assistant (EPBA)**. This system integrates structured clinical data (SQL) with unstructured medical reports (Vector DB) through a multi-agent orchestration layer.

---

## 1. Prerequisites

Ensure you have the following installed:

| Tool | Requirement | Purpose |
|------|-------------|---------|
| **Python** | 3.11+ | Backend & Agent logic |
| **Git** | Any | Source control |
| **Docker** | 20+ | Containerized deployment (Option B) |
| **Langfuse** | Account | Distributed observability & tracing |

### Required API Keys

Create a `.env` file in the project root:

```env
# OpenAI for LLM/Embeddings/Realtime API
OPENAI_API_KEY=sk-your-key-here

# Langfuse Configuration (Project: EPBA)
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_BASE_URL=https://cloud.langfuse.com # or your region
```

---

## 2. Installation

### Clone the Repository
```bash
git clone https://github.com/janmajay/EPBA
cd EPBA
```

### Virtual Environment & Dependencies
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e shared/ # Install shared library in editable mode
```

---

## 3. Data Ingestion

The system requires structured and unstructured data to be prepared before first run.

### 3.1 Ingest Structured Data (FHIR → SQLite)
Converts raw FHIR JSON bundles into a relational SQLite database.
```bash
python data/ingest_fhir_data.py --data_dir data/fhir_stu3
```

### 3.2 Ingest Unstructured Data (Reports → ChromaDB)
Process PDF/Text reports into the vector store.
```bash
python services/vector_agent/src/ingest.py
```

---

## 4. Running Locally (Development)

The quickest way to start all services for development:

```bash
python start_all_locally.py
```

**Services Started:**
- **Orchestrator** (`8000`): State orchestration (LangGraph)
- **SQL Agent** (`8001`): Optimized single-shot structured query
- **Vector Agent** (`8002`): Semantic report retrieval
- **Summarizer** (`8003`): Modular clinical synthesis
- **Agent Registry** (`8004`): Service discovery
- **Frontend** (`8501`): Streamlit UI

---

## 5. Running with Docker (Production-like)

```bash
make build   # Build images
make up      # Start containers
make logs    # Monitor orchestration
```

---

## 6. Observability (Langfuse)

EPBA is pre-instrumented for deep traceability. To view agent behaviors:
1. Log in to your [Langfuse Dashboard](https://cloud.langfuse.com).
2. Every request from the **Frontend** generates a root trace.
3. Drill down to see parallel **SQL Agent** generation vs **Vector Retrieval** latency and cost.

---

## 7. Troubleshooting

- **Port Conflicts**: `start_all_locally.py` automatically kills processes on ports 8000-8501.
- **Missing Data**: If queries return "No data found," ensure both ingestion scripts (Step 3) were run successfully.
- **Trace Missing**: Verify `LANGFUSE_SECRET_KEY` is correctly set in `.env` and matches the `cloudRegion: EU` configuration if applicable.

---

*Project maintained at: [github.com/janmajay/EPBA](https://github.com/janmajay/EPBA)*
