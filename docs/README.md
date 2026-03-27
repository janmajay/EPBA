# EPBA: Electronic Patient Bedside Assistant

**Hands-Free Healthcare Intelligence with Multi-Agent RAG.**

EPBA is a cutting-edge clinical assistant designed for high-stakes bedside environments. It bridges the gap between structured EHR data and unstructured medical reports through a voice-first, multi-agent orchestration layer.

---

## 🌟 Quick Links

- **[Main Overview](docs/README.md)**: Features, features, and quick-start.
- **[Implementation Details](docs/Implementation.md)**: Deep dive into the architecture, A2A protocol, and request lifecycle.
- **[Setup Guide](docs/Project_Setup.md)**: Detailed instructions for data ingestion, Docker, and troubleshooting.
- **[Use Case Presentation](docs/Use_Case_Presentation.md)**: Executive summary of clinical impact and human-AI interaction.

---

## 🚀 Getting Started

```bash
# Clone and install
git clone https://github.com/janmajay/EPBA
cd EPBA
pip install -r requirements.txt
pip install -e shared/

# Ingest data (Structured & Unstructured)
python data/ingest_fhir_data.py --data_dir data/fhir_stu3
python services/vector_agent/src/ingest.py

# Run
python start_all_locally.py
```

---

*Project maintained at: [github.com/janmajay/EPBA](https://github.com/janmajay/EPBA)*
