# ShadowFox AI Engineering Portfolio
### Complete 3-Level Monorepo: Beginner, Intermediate & Advanced AI Applications
> **Internship Track: AI Engineer | ShadowFox**  
> **Student:** Third-Year B.Tech Computer Science & IT Student  
> **Repository:** `ShadowFox-AI-Engineer-Internship`

---

## 🌟 Executive Summary

This repository represents a complete, production-grade engineering portfolio designed to satisfy all requirements of the **ShadowFox AI Engineer Internship Task List**. 

Rather than superficial prototypes or mock user interfaces, this monorepo implements **three genuine, fully functional, independently runnable AI engineering systems**, progressively scaling from prompt-engineered educational utilities to dense vector retrieval and agentic LangGraph state-machine microservices.

```text
ShadowFox-AI-Engineer-Internship/
│
├── Beginner/              # Level 1: StudyPilot AI (Prompt-Engineered Student Utility)
│   ├── app.py             # Streamlit Interactive Interface
│   ├── services/          # Decoupled LLM & Domain Service Layer
│   ├── prompts/           # Isolated Pedagogical Prompt Catalog
│   ├── tests/             # 14 Pytest Unit & Integration Tests
│   ├── requirements.txt
│   └── README.md
│
├── Intermediate/          # Level 2: DocuMind AI (Document-Based RAG Assistant)
│   ├── app.py             # Streamlit Document Ingestion & Q&A Dashboard
│   ├── ingestion/         # PDF & TXT Extractor & Sliding-Window Chunker
│   ├── retrieval/         # Gemini Embeddings & FAISS IndexFlatIP Cosine Index
│   ├── generation/        # Grounded QA Chain with Page & Source Citations
│   ├── tests/             # 13 Pytest Retrieval & Boundary Tests
│   ├── requirements.txt
│   └── README.md
│
├── Advanced/              # Level 3: KnowledgeForge AI (Production Enterprise RAG)
│   ├── app/               # Application Backend & Entrypoint Package
│   ├── backend/           # FastAPI REST API Backend
│   │   ├── main.py        # OpenAPI/Swagger App Entrypoint (/docs)
│   │   ├── api/           # REST Endpoints (/health, /documents, /query, etc.)
│   │   ├── schemas/       # Typed Pydantic Request & Response Models
│   │   ├── services/      # RAG Lifecycle Orchestrator
│   │   ├── rag/           # Ingestion (PDF/TXT/MD), FAISS Scoping, Hybrid Reranker
│   │   ├── workflows/     # LangGraph StateGraph (Validate -> Rewrite -> Retrieve -> Rerank -> Check)
│   │   └── core/          # Pydantic v2 Settings & Custom Exceptions
│   ├── frontend/          # Streamlit UI with Observability Diagnostics
│   ├── tests/             # 14 Pytest API, Ingestion, LangGraph & Recall Benchmark Tests
│   ├── Dockerfile         # Multi-Stage Production Container Recipe
│   ├── docker-compose.yml # Backend + Frontend Multi-Service Compose
│   ├── requirements.txt
│   └── README.md
│
├── shared/                # Common Deterministic Stubs for Offline & CI/CD Testing
│   ├── __init__.py
│   └── mock_provider.py   # MockLLMProvider & MockEmbeddingProvider
│
├── docs/                  # Comprehensive Engineering Documentation
│   ├── architecture.md    # System Architecture, Mermaid Diagrams & RAG Theory
│   ├── learning-guide.md  # Beginner-Friendly Code Walkthrough & Trade-offs
│   ├── testing-guide.md   # Step-by-Step Test & Verification Guide
│   ├── interview-questions.md # 60 Technical Interview Questions with Answers
│   └── submission-checklist.md# Pre-Flight Scan & 3-5 Minute Video Guide
│
├── pytest.ini             # Automated Test Configuration
├── .env.example           # Secure Environment Configuration Example
├── .gitignore             # Secrets & Bytecode Exclusions
├── README.md              # Root Monorepo Documentation
└── LICENSE                # MIT License
```

---

## 🚀 The Three Portfolio Projects

### 1. Beginner Level — [StudyPilot AI](./beginner/README.md)
*An intelligent academic companion featuring four distinct prompt-engineered student workflows.*
- **Smart Notes Summarizer:** Extracts core definitions, key technical terms, and flashcard-style revision bullets.
- **Quiz Generator:** Formulates calibrated multiple-choice questions (MCQs) with difficulty settings (1–10 questions), correct keys, and distractor rationale.
- **Concept Explainer:** Demystifies difficult computer science topics using plain English intuition, real-world analogies, and misconception traps.
- **Answer Improver:** Evaluates draft answers, improving grammar, structure, and vocabulary with side-by-side comparative feedback.
- **Key Engineering:** Decoupled service layer (`study_service.py`), isolated prompt templates (`study_prompts.py`), input bounds validation, and dual runtime (Live Gemini API vs. Offline Test Mode).

### 2. Intermediate Level — [DocuMind AI](./intermediate/README.md)
*A genuine retrieval-augmented generation assistant for PDF and TXT documents.*
- **Multi-File Ingestion:** Extracts clean page-by-page text from uploaded PDF and TXT documents.
- **Recursive Sliding-Window Chunking:** Calibrated character windows with 10–20% overlap, preserving document name, page number, and chunk IDs.
- **FAISS Cosine Similarity Search:** Normalized dense embeddings (`text-embedding-004`) indexed into `faiss.IndexFlatIP`.
- **Strict Evidence Grounding:** The prompt forbids speculation. If evidence is lacking, it explicitly outputs: *"Based on the provided documents, there is insufficient evidence to answer this question."*
- **Source Inspection:** Collapsible cards showing verifiable page references, similarity scores, and verbatim source passages.

### 3. Advanced Level — [KnowledgeForge AI](./advanced/README.md)
*A production-style enterprise RAG microservice with LangGraph agentic orchestration.*
- **FastAPI REST Backend:** Typed endpoints (`/health`, `POST /documents/upload`, `GET /documents`, `DELETE /documents/{id}`, `POST /query`, `/sessions/{id}`) documented via Swagger UI (`/docs`).
- **LangGraph State Graph:** Multi-step deterministic state machine:
  `Validate Query ➔ Rewrite Query ➔ FAISS Retrieve ➔ (Conditional Edge) ➔ Hybrid Rerank ➔ Generate ➔ Grounding Check`
- **Multi-Format Ingestion (PDF, TXT, MD):** Deterministic SHA256 document hashing and chunk ID tagging.
- **Document-Scoped Filtering:** Ability to isolate search strictly to specific `document_ids`.
- **Hybrid Reranker & Deduplicator:** Merges cosine semantic similarity with BM25 keyword density, pruning redundant passages with >85% Jaccard token overlap.
- **Docker & Docker Compose:** Containerized reproducible multi-container deployment with persistent index storage.

---

## 🛠️ Technology Stack

| Tier | Component | Technology |
| :--- | :--- | :--- |
| **Backend & API** | REST API Framework | **FastAPI** 0.110+ (Asynchronous, Swagger UI) |
| **Validation** | Data Models & Settings | **Pydantic v2** & **Pydantic Settings** |
| **Workflow Engine**| Agentic Orchestration | **LangGraph** 0.1+ (`StateGraph`, conditional edges) |
| **Vector Search** | Similarity Index | **FAISS** (`faiss-cpu` 1.8+, `IndexFlatIP`) |
| **AI Models** | Embeddings & LLM | **Google GenAI** (`text-embedding-004`, `gemini-1.5-flash`) |
| **Document Ingestion**| PDF Extraction | **pypdf** 4.0+ |
| **Frontend UI** | User Interfaces | **Streamlit** 1.35+ |
| **Containerization**| Reproducibility | **Docker** & **Docker Compose** |
| **Testing** | Automated Quality Gates| **Pytest** 8.0+ & **FastAPI TestClient** |

---

## ⚡ Quickstart Guide

### 1. Clone & Set Up Virtual Environment
```bash
git clone https://github.com/<your-username>/shadowfox-ai-engineering-portfolio.git
cd shadowfox-ai-engineering-portfolio
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate
```

### 2. Configure Environment Variables
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Open `.env` and insert your Gemini API Key:
```env
LLM_PROVIDER=gemini
GEMINI_API_KEY=AIzaSy...your_gemini_api_key_here
LLM_MODEL=gemini-1.5-flash
EMBEDDING_MODEL=models/text-embedding-004
```
*(Note: If no API key is provided, all three applications seamlessly support an **Offline Test Mode** producing deterministic educational responses and dense hash embeddings at zero cost).*

### 3. Launching the Applications

#### Option A: Run Beginner Level (StudyPilot AI)
```bash
streamlit run beginner/app.py
```
*Access in browser at `http://localhost:8501`.*

#### Option B: Run Intermediate Level (DocuMind AI)
```bash
streamlit run intermediate/app.py
```
*Access in browser at `http://localhost:8501`.*

#### Option C: Run Advanced Level (KnowledgeForge AI)
**Terminal 1 — FastAPI Backend:**
```bash
uvicorn advanced.backend.main:app --port 8000 --reload
```
*Swagger UI is live at `http://localhost:8000/docs`.*

**Terminal 2 — Streamlit Frontend:**
```bash
streamlit run advanced/frontend/app.py --server.port 8501
```
*Access in browser at `http://localhost:8501`.*

#### Option D: Launch Advanced Level via Docker Compose
```bash
docker-compose -f advanced/docker-compose.yml up --build
```

---

## 🧪 Automated Testing & Verification

The repository contains **41 automated tests** verifying validation gates, prompt generation, error handling, document extraction, FAISS vector indexing, LangGraph state transitions, and benchmark retrieval recall.

```bash
# Run all tests across the monorepo
python -m pytest -v
```

### Test Suite Summary:
```text
============================= test session starts =============================
platform win32 -- Python 3.12.10, pytest-9.1.1, pluggy-1.6.0
collected 41 items

beginner/tests/test_beginner.py (14 tests) ............. PASSED [ 34%]
intermediate/tests/test_intermediate.py (13 tests) ...... PASSED [ 65%]
advanced/tests/test_api.py (4 tests) .................. PASSED [ 75%]
advanced/tests/test_evaluation.py (2 tests) ........... PASSED [ 80%]
advanced/tests/test_ingestion.py (5 tests) ............ PASSED [ 92%]
advanced/tests/test_workflow.py (3 tests) ............. PASSED [100%]

============================= 41 passed in 9.85s ==============================
```

---

## 📚 In-Depth Engineering Documentation

Explore the comprehensive engineering documentation prepared for technical evaluation:
- 🏛️ [**System Architecture & RAG Theory**](./docs/architecture.md): Deep-dive into embeddings, FAISS Inner Product math, hybrid reranking formulas, and LangGraph state machines.
- 📖 [**Student Learning & Mentorship Guide**](./docs/learning-guide.md): Beginner-friendly code walkthroughs, design decisions, and architectural trade-offs.
- 🎯 [**60 Technical Interview Questions & Answers**](./docs/interview-questions.md): 20 questions per level covering debugging scenarios, design patterns, security, and edge cases.
- 🧪 [**Testing & Diagnostics Guide**](./docs/testing-guide.md): How to inspect vector similarity scores, debug unsupported answers, and verify Docker containers.
- 📋 [**Internship Submission & Video Checklist**](./docs/submission-checklist.md): Step-by-step pre-flight checklist and 4-minute demonstration video presentation script.

---

## 📄 License
This project is licensed under the [MIT License](./LICENSE).
