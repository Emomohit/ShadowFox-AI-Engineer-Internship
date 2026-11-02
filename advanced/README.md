# KnowledgeForge AI — Production-Style RAG Engineering Assistant
> **ShadowFox AI Engineering Internship — Level 3 (Advanced Track)**

---

## 1. Problem Statement
Simple RAG prototypes typically rely on linear scripts, fail silently under invalid payloads, lack document isolation, and cannot withstand production workloads. Real-world enterprise AI systems require:
- Strict **typed schema validation** (Pydantic).
- Explicit, inspectable **agentic workflow orchestration** with loop-bounds and fallback branches (**LangGraph**).
- Scalable, high-throughput REST API serving (**FastAPI**).
- Multi-format ingestion (**PDF, TXT, Markdown**) with stable cryptographic document IDs.
- **Document-scoped vector filtering** and **hybrid lexical-semantic reranking**.
- Containerized reproducibility via **Docker & Docker Compose**.

**KnowledgeForge AI** provides an end-to-end, production-style RAG architecture engineered to satisfy these real-world requirements.

---

## 2. Features and Scope
- **Multi-Format Ingestion:** Ingest PDF, TXT, and Markdown files up to 15 MB.
- **Cryptographic Hashing:** Uses SHA256 of file bytes to produce deterministic `document_id`s and stable chunk IDs (`{doc_id}_p{page}_c{idx}`).
- **Document-Scoped Filtering:** Query against the entire knowledge store or restrict retrieval strictly to specific `document_ids`.
- **LangGraph State Graph:** Multi-step cyclical-ready state machine:
  `Validate Query ➔ Rewrite Query ➔ FAISS Retrieve ➔ Hybrid Rerank ➔ Generate ➔ Grounding Check`
- **Conditional Fallbacks:** If 0 chunks match similarity criteria, branches deterministically to an `insufficient_evidence` node without wasting generation tokens.
- **Hybrid Reranker & Deduplicator:** Merges cosine similarity with query keyword overlap, and discards passages with >85% Jaccard token redundancy.
- **Production FastAPI Backend:** Complete with Swagger UI (`/docs`), centralized exception handling, and typed Pydantic models.
- **Modern Streamlit Frontend:** Clean reactive UI with document selection, chat cards, and full observability diagnostics.
- **Reproducible Containerization:** Complete multi-service `Dockerfile` and `docker-compose.yml`.

---

## 3. Technology Stack & Design Decisions
| Component | Technology | Rationale |
| :--- | :--- | :--- |
| **Backend REST API** | FastAPI 0.110+ | Asynchronous, high performance, automatic OpenAPI / Swagger documentation. |
| **Data Validation** | Pydantic v2 | Strict runtime type enforcement and schema serialisation. |
| **Workflow Engine** | LangGraph 0.1+ | StateGraph orchestration providing clear node transitions, conditional routing, and bounded retries. |
| **Vector Engine** | FAISS (`faiss-cpu`) | High-performance exact Inner Product (Cosine) similarity search. |
| **Embeddings & LLM**| `google-genai` | High-dimensional semantic vectors (`text-embedding-004`) and grounded generation (`gemini-1.5-flash`). |
| **Frontend UI** | Streamlit 1.35+ | Interactive dashboard with real-time diagnostics and document scoping. |
| **Containerization** | Docker & Docker Compose | Guaranteed reproducibility across operating systems. |
| **Testing** | Pytest & FastAPI TestClient | Comprehensive automated verification of API, ingestion, workflows, and evaluation metrics. |

---

## 4. Architecture Overview
```text
┌────────────────────────────────────────────────────────┐
│             Streamlit Frontend (app.py)                │
│    (Uploads, Document Scope Select, Chat, Diagnostics) │
└───────────────────────────┬────────────────────────────┘
                            │ HTTP / REST
                            ▼
┌────────────────────────────────────────────────────────┐
│             FastAPI Backend (main.py / routes.py)       │
│    - /health   - /documents/upload   - /documents      │
│    - /query    - /documents/{id}     - /sessions/{id}  │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│           RAGService (services/rag_service.py)         │
│   ┌────────────────────────────────────────────────┐   │
│   │               LangGraph Workflow               │   │
│   │ Validate ➔ Rewrite ➔ Retrieve ➔ Rerank ➔ Gen ➔ Check│
│   └───────────────────────┬────────────────────────┘   │
│                           │                            │
│         ┌─────────────────┴─────────────────┐          │
│         ▼                                   ▼          │
│  ┌──────────────┐                   ┌────────────────┐ │
│  │ FAISS Index  │                   │ Hybrid Rerank  │ │
│  │ (with Scope) │                   │ & Deduplication│ │
│  └──────────────┘                   └────────────────┘ │
└────────────────────────────────────────────────────────┘
```

---

## 5. End-to-End Workflow
1. **Upload:** Client sends multipart file to `POST /documents/upload`.
2. **Preprocessing:** `MultiFormatIngestor` cleans control characters, calculates SHA256 ID, and segments text into overlapping chunks.
3. **Indexing:** Chunks are embedded and stored in `DocumentScopedFAISSIndex`.
4. **Query Dispatched:** Client sends `POST /query` specifying question and optional document scope.
5. **LangGraph Traversal:**
   - `validate_query`: Ensures query length and constraints.
   - `rewrite_query`: Normalizes terminology.
   - `retrieve`: Vector search against FAISS filtered by allowed document IDs.
   - *Conditional Edge:* If chunks == 0, branch to `insufficient_evidence`.
   - `rerank`: Hybrid lexical-semantic scoring and Jaccard deduplication.
   - `generate`: Synthesizes answer using grounded system instructions.
   - `check_grounding`: Analyzes citation presence and calibrates confidence score.
6. **Delivery:** Typed `QueryResponse` returned to client with citations and diagnostics.

---

## 6. Installation Instructions
```bash
cd advanced
python -m pip install -r requirements.txt
```

---

## 7. Environment Variable Setup
Create `.env` with:
```env
LLM_PROVIDER=gemini
GEMINI_API_KEY=AIzaSy...your_gemini_key
LLM_MODEL=gemini-1.5-flash
EMBEDDING_MODEL=models/text-embedding-004
BACKEND_URL=http://localhost:8000
APP_ENV=development
```

---

## 8. Exact Commands to Run the Application
### Option A: Local Execution (FastAPI + Streamlit)
**Terminal 1 — Backend:**
```bash
uvicorn advanced.backend.main:app --host 0.0.0.0 --port 8000 --reload
```
Swagger UI will be live at `http://localhost:8000/docs`.

**Terminal 2 — Frontend:**
```bash
streamlit run advanced/frontend/app.py --server.port 8501
```
Open your browser at `http://localhost:8501`.

### Option B: Docker Compose
```bash
docker-compose -f advanced/docker-compose.yml up --build
```

---

## 9. API Specification & cURL Examples
### Health Check
```bash
curl -X GET "http://localhost:8000/health"
```

### Document Upload
```bash
curl -X POST "http://localhost:8000/documents/upload" \
  -F "file=@spec.md" \
  -F "chunk_size=500" \
  -F "chunk_overlap=50"
```

### Grounded Query
```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How does the hybrid reranker score candidates?",
    "top_k": 4,
    "similarity_threshold": 0.20,
    "enable_rerank": true
  }'
```

---

## 10. Observability & Diagnostics
Each query returns a rich `diagnostics` payload:
- `retrieved_count`: Number of chunks returned from raw FAISS search.
- `reranked_count`: Chunks remaining after hybrid reranking and deduplication.
- `grounding_score`: Calibrated confidence indicator (0.0 to 1.0).
- `grounding_reason`: Explanation of evidence support.
- `rewritten_query`: Normalized query executed in the vector space.

---

## 11. Troubleshooting
- **Port 8000 already in use:** Run backend on alternate port: `uvicorn advanced.backend.main:app --port 8080` and set `BACKEND_URL=http://localhost:8080`.
- **422 Validation Error on Query:** Ensure query string is at least 2 characters long.
- **Docker build error on Windows:** Ensure Docker Desktop is running and WSL2 is enabled.

---

## 12. Limitations & Future Improvements
- **Current Limitations:** In-memory FAISS index (persists during process lifetime; volume mounted in container); single node setup.
- **Future Enhancements:** Milvus or Qdrant distributed vector database cluster, asynchronous streaming tokens (SSE), and Redis session memory.

---

## 13. Testing & Evaluation Instructions
Run the entire advanced test suite covering API endpoints, ingestion, LangGraph transitions, and benchmark retrieval recall:
```bash
python -m pytest advanced/tests -v
```

---

## 14. Important Files & Directory Map
| File | Responsibility |
| :--- | :--- |
| `backend/main.py` | FastAPI application factory, CORS, and exception handlers. |
| `backend/api/routes.py` | REST API endpoints (`/health`, `/documents`, `/query`, etc.). |
| `backend/schemas/models.py` | Pydantic typed request and response schemas. |
| `backend/services/rag_service.py` | High-level orchestrator coordinating indexing and querying. |
| `backend/rag/ingestion.py` | Multi-format parser (PDF, TXT, MD) with SHA256 chunking. |
| `backend/rag/embeddings.py` | Production Gemini and mock vector embeddings driver. |
| `backend/rag/vector_index.py` | DocumentScopedFAISSIndex supporting document filtering. |
| `backend/rag/reranker.py` | Hybrid lexical-semantic scoring and Jaccard deduplication. |
| `backend/workflows/graph.py` | LangGraph StateGraph definition and node routing. |
| `backend/workflows/states.py` | Typed RAGState structure. |
| `frontend/app.py` | Production Streamlit frontend connecting to FastAPI backend. |
| `Dockerfile` | Multi-stage container recipe. |
| `docker-compose.yml` | Multi-container composition (Backend + Frontend). |
| `tests/` | Comprehensive pytest suite including benchmark evaluation. |
