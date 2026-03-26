# EPBA Project Presentation Content

**Duration:** 20 Minutes
**Format:** 7 content slides + 1 Reference slide

---

## Slide 1: Problem Motivation
**Title:** The "Information Silo" Bottleneck in Healthcare
* **Cognitive Overload:** Doctors transition constantly between examining patients and "chart hunting" across multiple platforms, distracting from the bedside interaction.
* **Fragmented Data:** Crucial patient information is split between structured EMR databases (vitals, labs) and unstructured PDF reports (imaging, discharge summaries).
* **Time & Mobility Constraints:** Reviewing comprehensive files on stationary terminals is slow, interrupts workflow, and increases hygiene/infection risks due to handling physical devices.
* **The Solution (EPBA):** A voice-first, multi-agent Retrieval-Augmented Generation (RAG) system providing instant, hands-free access to unified patient records.

---

## Slide 2: Data Source and Preparation
**Title:** Unifying Structured and Unstructured Clinical Data
* **Structured Data Ingestion (SQL):** 
  * **Source:** Raw FHIR STU3 JSON bundles (~557 files).
  * **Preparation:** Extracted and mapped into 9 normalized SQLite database tables (Patients, Encounters, Conditions, Medications, Observations, Allergies, CarePlans, Immunizations, Procedures).
* **Unstructured Data Ingestion (Vector):**
  * **Source:** Medical reports in PDF and TXT formats.
  * **Preparation:** Loaded via PyPDFLoader, chunked using `RecursiveCharacterTextSplitter` (chunk size: 600, overlap: 50) to retain semantic context.
  * **Storage:** Embedded and persisted in ChromaDB for fast similarity search.

---

## Slide 3: Variable Selection
**Title:** Feature Engineering and Context Retrieval Selection
* **Structured Schema Selection:** Rather than selecting raw ML variables, EPBA selectively injects the full schema of the 9 patient-centric tables into the SQL Agent's prompt to ensure precise, zero-shot SQL generation for targeted query parameters (e.g., patient ID, specific vitals).
* **Vector Retrieval Parameters:** Selected `k=3` top most relevant document chunks to feed into the summarizer, balancing context richness with context-window limits.
* **Dynamic Content Extraction:** Variables synthesized dynamically based on clinician queries—extracting specific demographic details, recent medical procedures, and conflicting named entities across both data modalities.

---

## Slide 4: Model and Results - System Architecture
**Title:** Parallel Multi-Agent Orchestration
* **Dual-Agent Execution:** The Orchestrator routes the query concurrently to the SQL Agent (structured fast lookup) and Vector Agent (semantic document search) using the Google A2A protocol.
* **Model Choices:** 
  * OpenAI `gpt-4o-mini` handles complex context synthesis.
  * `text-embedding-3-small` handles embeddings.
  * `Whisper-1` & GPT-Realtime processes low-latency audio inputs and outputs.

![EPBA High-Level Architecture](diagrams/architecture.png)

---

## Slide 5: Model and Results - Summarization
**Title:** Modality-Aware Clinical Synthesis
* **Decision Flow:** The final step dynamically branches based on the doctor's input medium.
* **Text Queries:** GPT-4o-mini provides comprehensive markdown responses with citations, flagging discordant information.
* **Voice Queries:** GPT-Realtime interprets the combined context to deliver a rapid, spoken-style summary suitable for natural bedside listening.
* **Results:** End-to-end execution of parallel database and vector lookups followed by intelligent synthesis takes <12 seconds.

![Summarizer Data Flow](diagrams/summarizer_flow.png)

---

## Slide 6: Evaluation Metrics and Reasoning
**Title:** Ensuring Clinical Safety and Accuracy
* **RAG Retrieval Accuracy (DeepEval):**
  * **Contextual Relevancy:** Measures if retrieved SQL rows and PDF chunks directly address the doctor's query, reducing noise.
  * **Faithfulness:** Ensures the summarization agent's output is derived *strictly* from the retrieved data (zero hallucination tolerance).
  * **Answer Relevance:** Validates that the final multimodal output directly answers the specific medical question.
* **Traceability & Observability:** Integrated **Langfuse Distributed Tracing** continuously monitors every inference span, token usage, and source attribution for auditing and safety.

---

## Slide 7: Business Implication
**Title:** Strategic Impact on Healthcare Delivery
* **Efficiency & Throughput:** Removes 2-4 minutes of manual data retrieval per patient, significantly expanding physician capacity and focus time.
* **Enhanced Bedside Experience:** Doctors maintain eye contact and hands-free interaction; data follows the clinician directly through an "Aura of Intelligence."
* **Clinical Safety & Compliance:** Tracing ensures every AI-generated answer is verifiable against a specific SQL row or PDF paragraph, meeting high-stakes clinical accountability standards.
* **Scalability:** Dockerized, microservices architecture enables easy deployment and scaling to different hospital departments and varying medical datasets.

---

## Slide 8: References
* **EPBA Patient Synthetic Data Generator:** https://synthea.mitre.org/downloads
* **Framework Inspiration:** *Building Applications with AI Agents* — O'Reilly Media
