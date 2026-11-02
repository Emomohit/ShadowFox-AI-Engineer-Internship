# Comprehensive Testing & Verification Guide
## ShadowFox AI Engineering Portfolio

This guide details how to verify each application, run automated pytest suites, inspect vector retrieval metrics, diagnose edge cases, and test containerized deployments.

---

## 1. Quick Automated Test Execution

You can run the full test suite (41 tests across all 3 levels) from the repository root:

```bash
# Run all tests across the entire monorepo
python -m pytest -v
```

To run individual project test suites:
```bash
# Beginner Level (14 tests)
python -m pytest beginner/tests -v

# Intermediate Level (13 tests)
python -m pytest intermediate/tests -v

# Advanced Level (14 tests)
python -m pytest advanced/tests -v
```

---

## 2. Testing Level 1: StudyPilot AI

### A. Manual Functional Verification
1. Launch the app: `streamlit run beginner/app.py`
2. **Notes Summarizer:** Click *"📥 Load Sample Operating Systems Notes"*, select *"Detailed Summary"*, and click *"🚀 Generate Summary"*. Verify that key definitions (Virtual Memory, Paging, TLB) and bullet summaries render cleanly.
3. **Quiz Generator:** Click *"📥 Load Sample Algorithms Notes"*, select 3 questions at "Medium" difficulty, and generate. Verify that options A, B, C, D, highlighted correct answers, and explanations are displayed.
4. **Concept Explainer:** Click *"📥 Load Example Topic"* ("How Database Indexing works") and verify the intuition, real-world analogy, and common misconception sections.
5. **Answer Improver:** Click *"📥 Load Sample Student Answer"* and verify that comparative feedback and the polished draft appear side-by-side.

### B. Error Handling & Edge Cases
- **Empty Input:** Click "Generate Summary" with an empty text box. Verify that an informative error message (`⚠️ Study Notes cannot be empty`) is displayed.
- **Short Input (<10 chars):** Type "Hello" and submit. Verify: `⚠️ Study Notes is too short (5 chars). Minimum required is 10 characters.`
- **Offline / Test Mode:** Toggle the sidebar switch to *"🧪 Demo / Offline Test Mode"*. Confirm the badge changes to yellow and features generate deterministic educational responses without an API key.

---

## 3. Testing Level 2: DocuMind AI

### A. Manual Ingestion & RAG Verification
1. Launch the app: `streamlit run intermediate/app.py`
2. In the *"Document Upload"* tab, click *"📥 Load Sample Architecture Overview (TXT)"*.
3. Verify the status message: `Successfully indexed 'ShadowFox_RAG_Architecture.txt' (3 chunks in FAISS index)!`
4. Switch to the *"Grounded Question Answering"* tab.
5. Click *"💡 Load Sample Query"* (*"How does the ingestion pipeline handle chunk boundaries and overlap?"*).
6. Click *"🔍 Search & Answer"*.
7. Verify:
   - Grounded answer text citing the 10% overlap specification.
   - Collapsible *"Supporting Evidence & Citations"* box showing document name, page number, and similarity score.
8. Switch to *"Knowledge Base Inspector"* tab and verify the raw text of each indexed chunk.

### B. Testing Edge Cases & Grounding Boundaries
- **Unrelated Out-of-Domain Query:** Ask *"What is the recipe for baking chocolate cake?"*
  - **Expected Result:** The model returns: *"Based on the provided documents, there is insufficient evidence to answer this question."*
- **Empty Query:** Click "Search & Answer" with a blank input. Verify validation alert.
- **Document Removal:** Click the 🗑️ trash icon next to the document in the sidebar. Confirm chunks are wiped from the store.

---

## 4. Testing Level 3: KnowledgeForge AI

### A. Backend Health & API Verification
1. Start the FastAPI backend:
   ```bash
   uvicorn advanced.backend.main:app --port 8000 --reload
   ```
2. Open Swagger UI at `http://localhost:8000/docs`.
3. Execute `GET /health` and confirm:
   ```json
   {
     "status": "ok",
     "version": "1.0.0",
     "provider": "gemini",
     "environment": "development"
   }
   ```

### B. Ingesting Documents via API
```bash
curl -X POST "http://localhost:8000/documents/upload" \
  -F "file=@spec.md" \
  -F "chunk_size=500" \
  -F "chunk_overlap=50"
```
Verify response status is `201 Created` with a 16-character hexadecimal `document_id`.

### C. Testing Grounded Query via cURL
```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the core stages in the LangGraph workflow?",
    "top_k": 3,
    "similarity_threshold": 0.05,
    "enable_rerank": true
  }'
```
Verify the returned JSON contains `answer`, `grounded: true`, `sources`, and `diagnostics`.

### D. Testing the Streamlit Frontend
Launch the frontend:
```bash
streamlit run advanced/frontend/app.py --server.port 8501
```
Verify the sidebar shows green *"● Connected to FastAPI (http://localhost:8000)"*.

---

## 5. Verifying Docker & Container Execution

If Docker and Docker Compose are installed on your workstation:
```bash
# Build and launch both Backend and Frontend containers
docker-compose -f advanced/docker-compose.yml up --build

# Verify running containers
docker ps

# Check backend container health
curl -f http://localhost:8000/health

# Access frontend in browser
http://localhost:8501
```
To tear down:
```bash
docker-compose -f advanced/docker-compose.yml down
```

---

## 6. How to Diagnose Inaccurate or Unsupported Answers

If a query returns an unexpected answer:
1. **Check Retrieval Similarity Scores:** In DocuMind AI or KnowledgeForge AI, expand the *"Supporting Evidence"* or *"Diagnostics"* tab. If scores are below 0.20, the chunks are semantically distant from the question.
2. **Inspect Chunk Boundaries:** In the Knowledge Base Inspector, check if critical facts were split awkwardly across chunk boundaries. If so, increase `chunk_size` or `chunk_overlap`.
3. **Check Lexical Density:** If the hybrid reranker is enabled, confirm whether the query keywords appear verbatim in the source document.
