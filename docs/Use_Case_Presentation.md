# EPBA: Electronic Patient Bedside Assistant
**Healthcare Intelligence at the Speed of Conversation**

---

## Slide 1: Executive Summary

**Concept**:
EPBA is a voice-first **Retrieval-Augmented Generation (RAG)** system that provides instant access to patient data via **Speech-to-Speech (S2S)** and Text modalities.

**The Vision**:
Eliminate "Chart Hunting." Doctors transition from searching for data to **talking to the medical record**, leaving hands free for physical exams and eyes focused on the patient.

---

## Slide 2: The Practical Problem

**The "Information Silo" Bottleneck**:
*   **Structured vs Unstructured**: Crucial data is split between EMR databases (vitals, labs) and PDF reports (imaging, discharge summaries).
*   **Cognitive Load**: Shuffling through papers or scrolling tablets distracts from the bedside interaction.
*   **Hygiene & Mobility**: Handling physical files or stationary terminals is slow and increases infection risk.

---

## Slide 3: Intelligent Architecture

EPBA uses a **Parallel Multi-Agent Orchestration** layer to bridge the data gap:

1.  **Dual-Modality Processing**: 
    -   **Text**: High-fidelity chat for detailed clinical review.
    -   **Voice**: Low-latency S2S via OpenAI Realtime API for hands-free queries.
2.  **Multi-Agent Retrieval (A2A Protocol)**:
    -   **SQL Agent**: Generates single-shot queries over structured EHR data.
    -   **Vector Agent**: Performs semantic search over medical PDF reports.
3.  **Clinical Synthesis**: Summarizer agent fuses structured and unstructured context into a balanced clinical answer.
4.  **Observability**: Integrated **Langfuse Distributed Tracing** for "Black Box" transparency—monitor every inference for safety and accuracy.

---

## Slide 4: Real-World Clinical Flow

**Scenario: Rapid Bedside Evaluation**

1.  **Voice Query**: Dr. Smith asks, *"EPBA, did this patient's last MRI mention a meniscus tear, and what are their current vitals?"*
2.  **Parallel Search**: The Orchestrator triggers SQL (vitals) and Vector (MRI PDF) agents simultaneously.
3.  **Synthesis**: Data is aggregated; the Summarizer generates a concise clinical insight.
4.  **Multimodal Response**:
    -   **Audio**: "Yes, MRI from June confirms a lateral meniscus tear. Current BP is 120/80."
    -   **Screen**: Side-by-side view of the specific MRI excerpt and the latest vitals chart.

---

## Slide 5: Strategic Impact

**1. Efficiency & Throughput**:
Removes 2-4 minutes of data retrieval per patient, allowing for more focused care.

**2. Clinical Safety**:
Langfuse tracing ensures every AI answer is traceable to a specific source (SQL row or PDF paragraph).

**3. Enhanced Bedside Experience**:
Physical charts are replaced by an "Aura of Intelligence"—data follows the doctor, not the other way around.

---

*Explore the Code: [github.com/janmajay/EPBA](https://github.com/janmajay/EPBA)*
