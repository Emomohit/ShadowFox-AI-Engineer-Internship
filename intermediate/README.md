# DocuMind AI — Document-Based Question Answering Assistant
> **ShadowFox AI Engineering Internship — Level 2 (Intermediate Track)**

---

## 1. Problem Statement
Foundation Large Language Models (LLMs) suffer from two fundamental weaknesses in enterprise and academic settings:
1. **Knowledge Cutoffs & Parametric Amnesia:** They cannot answer queries regarding private, local, or newly published documents.
2. **Hallucination:** When asked about unknown specifics, they tend to fabricate believable but completely false answers.

**DocuMind AI** solves this by implementing a genuine **Retrieval-Augmented Generation (RAG)** pipeline. Users can upload arbitrary PDF and TXT documents. The system decomposes them into indexed vector embeddings, retrieves the top-k most semantically relevant text passages, and forces the language model to synthesize answers strictly substantiated by retrieved evidence—citing page numbers and document names directly.

---

## 2. Features and Scope
- **Multi-File Ingestion:** Ingest multiple PDF and TXT documents into a unified knowledge session.
- **Robust Extraction:** Handles text encoding fallbacks (UTF-8 / Latin-1) and detects empty, corrupt, or unscanned pages.
- **Recursive Chunking:** Configurable sliding window with chunk size (150–1200 characters) and overlap (0–200 characters).
- **Metadata Traceability:** Every chunk preserves `source`, `page_number`, `chunk_id`, and `token_estimate`.
- **Dense Vector Search:** Uses **FAISS IndexFlatIP** over L2-normalized embeddings, guaranteeing exact cosine similarity computation.
- **Grounding-Constrained Synthesis:** The prompt strictly forbids ungrounded hallucination. If evidence is lacking, it explicitly yields: *"Based on the provided documents, there is insufficient evidence to answer this question."*
- **Source Inspection & Verification:** Collapsible evidence cards display the exact source document, page number, similarity score, and verbatim excerpt.
- **Active Document Management:** Inspect indexed chunks, delete individual documents, or wipe the vector store cleanly.
- **Dual Runtime Support:** Switch between live **Google Gemini API** (`text-embedding-004` & `gemini-1.5-flash`) and an **Offline / Deterministic Test Mode**.

---

## 3. Technology Stack & Design Decisions
| Component | Technology | Rationale |
| :--- | :--- | :--- |
| **Language** | Python 3.12 | Standard AI engineering environment. |
| **UI Framework** | Streamlit 1.35+ | Rapid, state-driven dashboard for document uploads and interactive chat. |
| **PDF Extraction** | `pypdf` 4.0+ | Pure Python, lightweight, reliable page-by-page text and metadata parser. |
| **Vector Index** | `faiss-cpu` 1.8+ | Industry-standard high-performance similarity search library developed by Meta AI. |
| **Embeddings & LLM** | `google-genai` | High-dimensional dense vectors (`text-embedding-004`, 768-dim) and fast generative inference (`gemini-1.5-flash`). |
| **Array Mathematics**| `numpy` 1.26+ | Fast L2 normalization for vector cosine equivalence. |
| **Testing** | `pytest` | Comprehensive unit and integration test coverage. |

---

## 4. Architecture Overview
```text
┌────────────────────────────────────────────────────────┐
│               Streamlit UI Layer (app.py)              │
│    (File Uploader, Knowledge Inspector, Q&A View)      │
└───────────────────────────┬────────────────────────────┘
                            │
               ┌────────────┴────────────┐
               ▼                         ▼
┌─────────────────────────────┐ ┌────────────────────────────────┐
│   Ingestion Pipeline        │ │      Retrieval Pipeline        │
│  - loader.py (PDF / TXT)    │ │  - embeddings.py (Gemini/Mock) │
│  - chunker.py (Sliding Win) │ │  - vector_store.py (FAISS)     │
└──────────────┬──────────────┘ └────────────────┬───────────────┘
               │                                 │
               └───────────────┬─────────────────┘
                               ▼
┌────────────────────────────────────────────────────────────────┐
│        Generation Pipeline (qa_chain.py)                       │
│  - Grounded System Prompt                                      │
│  - Context-Limited Evidence Package                            │
│  - Verifiable Page & Source Citations                          │
└────────────────────────────────────────────────────────────────┘
```

---

## 5. End-to-End Workflow
1. **Upload & Extract:** User uploads PDF or TXT files. `DocumentLoader` extracts text and isolates page numbers.
2. **Recursive Chunking:** `TextChunker` segments pages into overlapping windows while attaching metadata tags.
3. **Embedding:** `EmbeddingService` converts chunks into dense vector embeddings.
4. **FAISS Indexing:** Embeddings are normalized and inserted into `FAISSVectorStore` (`IndexFlatIP`).
5. **Query Embedding & Retrieval:** When a user queries, the question is embedded and compared against the index to retrieve the top-K chunks exceeding the similarity threshold.
6. **Grounded Generation:** `GroundedQAChain` binds retrieved context into an evidence block and prompts the LLM to formulate an answer with citations.
7. **Display:** Answer and verifiable source cards are rendered to the user.

---

## 6. Installation Instructions
```bash
cd intermediate
python -m pip install -r requirements.txt
```

---

## 7. Environment Variable Setup
Ensure your `.env` contains:
```env
LLM_PROVIDER=gemini
GEMINI_API_KEY=AIzaSy...your_gemini_key
LLM_MODEL=gemini-1.5-flash
EMBEDDING_MODEL=models/text-embedding-004
```
*If no API key is supplied, toggle "🧪 Demo / Offline Test Mode" in the sidebar to run with deterministic dense vector hashing and local mock responses.*

---

## 8. Exact Commands to Run the Application
From the repository root:
```bash
streamlit run intermediate/app.py
```
Or from within `intermediate/`:
```bash
streamlit run app.py
```
Open `http://localhost:8501`.

---

## 9. Example Inputs & Expected Outputs
### Document Ingestion
- Upload `ShadowFox_RAG_Architecture.txt` (or click "Load Sample Architecture Overview").
- System indexes 3 chunks into the FAISS store.

### User Query
- **Question:** *"How does the ingestion pipeline handle chunk boundaries and overlap?"*
- **Expected Answer:**
  > *"The document ingestion pipeline segments content into semantically coherent chunks calibrated between 400 and 800 characters, incorporating a 10% overlap to preserve boundary context across segments."*
- **Source Citation:**
  > `📄 ShadowFox_RAG_Architecture.txt — Page 1 | Similarity Score: 0.82`

---

## 10. API Usage & Quota Guidelines
- **Embeddings:** Free tier allows 1,500 Requests Per Minute (RPM) for `text-embedding-004`.
- **Chat:** 15 Requests Per Minute (RPM) for `gemini-1.5-flash`.
- Documents are embedded in client memory; no third-party vector databases or paid subscriptions are required.

---

## 11. Troubleshooting
- **Scanned PDF error:** If you see *"PDF appears to be a scanned image"*, the PDF does not contain selectable text layers (OCR is required). Use clean digital PDFs or text files.
- **No relevant passages found:** Lower the *Min Similarity Threshold* slider in the sidebar (e.g. from 0.4 to 0.1) or increase *Top-K Retrieved Chunks*.
- **FAISS Import Issue on Windows:** Make sure `faiss-cpu` is installed, not the GPU variant.

---

## 12. Limitations & Future Improvements
- **Current Limitations:** Session-scoped in-memory FAISS store (resets on application restart unless persisted to disk); does not support tabular OCR or images.
- **Future Enhancements:** Hybrid BM25 + dense retrieval, persistent disk serialization, and cross-encoder neural reranking (implemented in Level 3).

---

## 13. Testing Instructions
Run the automated test suite testing loader, chunker, FAISS store, and grounded response logic:
```bash
python -m pytest intermediate/tests -v
```

---

## 14. Important Files & Directory Map
| File | Responsibility |
| :--- | :--- |
| `app.py` | Streamlit UI with file uploader, RAG controls, and chat interface. |
| `ingestion/loader.py` | Extracts text and page numbers from PDF and TXT files. |
| `ingestion/chunker.py` | Sliding window chunker with metadata tagging. |
| `retrieval/embeddings.py` | Embedding generation driver (Gemini + local mock). |
| `retrieval/vector_store.py`| FAISS `IndexFlatIP` wrapper for cosine similarity search and doc deletion. |
| `generation/qa_chain.py` | Grounded prompt assembly, citation structuring, and insufficient evidence detection. |
| `tests/test_intermediate.py` | Pytest test suite covering RAG edge cases. |
| `requirements.txt` | Intermediate level dependencies. |
