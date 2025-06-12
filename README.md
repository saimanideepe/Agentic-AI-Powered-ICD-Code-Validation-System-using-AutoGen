# 🏥 ICD-10 Code Extraction and Validation with Multi-LLM RAG

This project provides a powerful solution for **extracting, validating, and scoring ICD-10 codes** from medical text summaries. It integrates outputs from multiple Large Language Models (LLMs) using a **Retrieval Augmented Generation (RAG)** pipeline — leveraging OpenAI's `gpt-4o` and Groq-hosted models like **Mistral** and **LLaMA**. The system performs end-to-end medical coding, validation, and explanation in a structured, automated manner.

## 🧱 Architecture

The system follows a modular pipeline involving:

1. Medical summary ingestion (from RAG)
2. ICD-10 extraction by LLMs
3. Validation and refinement
4. Confidence scoring with evidence
5. Final output schema generation

![Architecture Diagram](Screenshot%202025-06-13%20031449.png)

---
## ✨ Features

- **🔗 Multi-LLM Integration**  
  Combines outputs from multiple LLMs to ensure diverse reasoning and improve ICD-10 prediction accuracy.

- **🧠 Intelligent ICD-10 Code Extraction**  
  Extracts potential ICD-10 codes from unstructured LLM responses using prompt engineering and model filtering.

- **✅ LLM-Powered Validation & Refinement**  
  Validates the correctness of each ICD-10 code in the given medical context, with retry prompts for better alternatives.

- **📊 Confidence Scoring**  
  Assigns a granular confidence score (0–100) to each ICD-10 code using LLM judgment and summary-backed evidence.

- **🔍 Contextual Evidence Extraction**  
  Extracts relevant sentences from the medical text to justify the inclusion of each code.

- **📦 Structured Output**  
  Produces a detailed, LLM-tagged JSON schema for easy downstream usage or integration into medical systems.

- **⚙️ Automated Workflow**  
  Handles all steps from raw input to final validated schema generation across all models.

---

## 🚀 Installation

To run this project locally:

### To Run

```bash
git clone <repository-url>
cd <repository-directory>
python agent.py


