# EPBA: Electronic Patient Bedside Assistant

**Hands-Free Healthcare Intelligence with Multi-Agent RAG.**

EPBA is a cutting-edge clinical assistant designed for high-stakes bedside environments. It bridges the gap between structured EHR data and unstructured medical reports through a voice-first, multi-agent orchestration layer.

---

## 🌟 Key Features

- **🗣️ Speech-to-Speech (S2S)**: Powered by OpenAI Realtime API for ultra-low latency, hands-free voice interactions.
- **🤖 Multi-Agent Orchestration**: Utilizes the **Google A2A Protocol** to coordinate specialized agents:
  - **SQL Agent**: High-performance structured EHR querying (0-shot generation).
  - **Vector Agent**: Semantic retrieval over unstructured PDF medical reports (ChromaDB).
  - **Summarization Agent**: Intelligent clinical synthesis of disparate data points.
- **🕵️ Distributed Traceability**: End-to-end observability and safety monitoring via **Langfuse**.
- **💻 Hybrid Architecture**: Comprehensive support for both Local Development and Docker-based deployment.

---

## 🏗️ Project Structure

```text
EPBA/
├── services/
│   ├── orchestrator/      # State management & agent routing (LangGraph)
│   ├── sql_agent/         # Structured data retrieval (SQLAlchemy)
│   ├── vector_agent/      # Unstructured report RAG (ChromaDB)
│   ├── summarization_agent/ # Clinical synthesis engine
│   ├── agent_registry/    # Service discovery via A2A protocol
│   └── frontend/          # Multi-modal Streamlit UI (Voice + Text)
├── shared/                # Shared internal logic, models, and config
├── config/                # Centralized YAML configuration
├── data/                  # Clinical databases, FHIR bundles, and PDF reports
├── docs/                  # In-depth technical & use-case documentation
└── docker-compose.yml     # Production-grade container orchestration
```

---

## 🚀 Quick Start

### 1. Installation
The fastest way to get started is to follow the [Project Setup Guide](docs/Project_Setup.md):
```bash
git clone https://github.com/janmajay/EPBA
cd EPBA
pip install -r requirements.txt
pip install -e shared/
```

### 2. Configure Environment
Create a `.env` file with your keys:
```env
OPENAI_API_KEY=sk-...
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
```

### 3. Run All Services
```bash
python start_all_locally.py
```

---

## 📚 Deep Dives

- **[Implementation Details](docs/Implementation.md)**: Deep dive into the architecture, A2A protocol, and request lifecycle.
- **[Use Case Presentation](docs/Use_Case_Presentation.md)**: Executive summary of clinical impact and human-AI interaction.
- **[Setup Guide](docs/Project_Setup.md)**: Detailed instructions for data ingestion, Docker, and troubleshooting.

---

## 🛡️ Clinical Traceability

EPBA treats AI safety as a first-class citizen. Every query is tracked via **Langfuse**, allowing clinicians to trace summarized insights back to the original SQL record or PDF paragraph.

---

*Maintained at: [github.com/janmajay/EPBA](https://github.com/janmajay/EPBA)*
