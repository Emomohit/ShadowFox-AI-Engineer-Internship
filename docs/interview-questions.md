# Technical Interview Question & Answer Bank
## ShadowFox AI Engineering Internship Evaluation

This document contains **60 technical interview questions (20 per level)** with detailed, engineering-grade answers covering architecture, design trade-offs, debugging scenarios, security, and edge-case handling.

---

# PART 1: BEGINNER LEVEL — StudyPilot AI (20 Questions)

### Q1. Why did you separate prompts into a dedicated `prompts/` module instead of inlining them in `app.py`?
**Answer:** Inlining prompts directly into UI controllers violates the Single Responsibility Principle. Separating prompt templates decouples prompt engineering and prompt versioning from the presentation layer. It allows developers to test, benchmark, and update system prompts without modifying UI logic or risking frontend regressions.

### Q2. How does StudyPilot AI validate student inputs before making an LLM API call?
**Answer:** `StudyService` enforces three validation gates: (1) Non-empty checks stripping whitespace, (2) Minimum character threshold (>=10 chars) to prevent wasteful calls on single characters or noise, and (3) Maximum character threshold (<=15,000 chars) to protect the LLM context window and avoid excessive token billing.

### Q3. How does the application prevent prompt injection attacks when a student pastes malicious notes?
**Answer:** User text is never directly concatenated into instructions. It is cleanly demarcated inside explicit boundary delimiters (e.g. `\"\"\"STUDENT NOTES...\"\"\"`) accompanied by clear system instructions commanding the model to treat the delimited payload as passive text to analyze, rather than executable instructions.

### Q4. How do you handle HTTP 429 (Rate Limit / Resource Exhausted) errors from the Gemini API?
**Answer:** In `llm_service.py`, exceptions from the provider client are caught. If error strings contain `RESOURCE_EXHAUSTED` or `429`, the service translates them into an informative, user-friendly `LLMServiceError` advising the student to pause briefly or verify their billing quota, preventing unhandled application crashes.

### Q5. What is the role of the `MockLLMProvider` in your beginner application?
**Answer:** The mock provider enables deterministic automated testing in CI/CD pipelines and allows reviewers to evaluate all UI features offline without consuming API tokens or requiring private API keys.

### Q6. How do you ensure API keys are not accidentally exposed in source code?
**Answer:** API keys are never hardcoded. They are loaded at runtime from `.env` via `python-dotenv`. Furthermore, `.env` and `*.key` files are explicitly added to `.gitignore`.

### Q7. Why did you choose Google Gemini 1.5 Flash over larger frontier models?
**Answer:** Gemini 1.5 Flash provides sub-second latency, generous free-tier quotas (15 RPM), a massive context window, and adequate reasoning ability for summarization, quiz generation, and grammar refinement, making it the most cost-effective choice for a student utility app.

### Q8. In the Quiz Generator, how do you prevent the model from generating questions on unrelated topics?
**Answer:** The system prompt explicitly instructs: *"Generate questions strictly based on the provided study notes. Do not reference outside topics not present or implied by the notes."*

### Q9. What happens if a student selects a quiz question count outside the 1–10 range?
**Answer:** `StudyService.generate_quiz()` performs boundary validation and raises a `ValueError("Number of questions must be between 1 and 10.")` before dispatching any API call.

### Q10. Why is temperature set to 0.3 rather than 0.7 or 1.0 in `LLMService`?
**Answer:** Educational summarization and quiz generation require deterministic, factual consistency and strict structural compliance. A low temperature of 0.3 minimizes randomness and reduces hallucination risk.

### Q11. How does the Answer Improver provide constructive feedback rather than just rewriting?
**Answer:** The prompt template instructs the model to produce a comparative breakdown divided into: (1) Polished version, (2) Specific vocabulary and structural improvements made, and (3) Actionable tips for future exam writing.

### Q12. What is the difference between client-side validation and domain-service validation?
**Answer:** Client-side validation (in UI) provides instant user feedback (e.g. disabling buttons if text is blank). Domain-service validation (in `study_service.py`) protects the core business logic and API layer regardless of whether requests arrive from Streamlit, CLI scripts, or automated tests.

### Q13. How would you debug an issue where the model generates an empty response?
**Answer:** Empty responses typically occur when Gemini's safety filters flag input or output tokens (e.g., medical or sensitive terms). We inspect the response's `finish_reason` and safety ratings. In `llm_service.py`, if `response.text` is empty, we raise a diagnostic exception indicating potential safety filter activation.

### Q14. What design pattern was applied in `llm_service.py`?
**Answer:** The **Adapter / Gateway Pattern**. It wraps third-party SDK calls behind a unified `generate_text()` interface, shielding the rest of the application from provider-specific breaking changes.

### Q15. How would you measure the readability score of the generated summaries?
**Answer:** We could integrate standard algorithmic readability metrics such as the Flesch-Kincaid Grade Level or Coleman-Liau Index, calculating sentence length and syllable count on the generated text.

### Q16. Can StudyPilot AI support other providers like Anthropic Claude or OpenAI?
**Answer:** Yes. Because `LLMService` encapsulates provider initialization, adding OpenAI simply involves initializing `openai.OpenAI()` when `LLM_PROVIDER=openai` without modifying any UI or prompt files.

### Q17. How do you test prompt changes without incurring API costs?
**Answer:** Unit tests verify that prompt templates correctly interpolate variables, while integration tests execute against `MockLLMProvider`, asserting output formats and error conditions at zero cost.

### Q18. What is the trade-off of running Streamlit compared to a React/FastAPI architecture?
**Answer:** Streamlit enables rapid full-stack Python development without writing JavaScript or REST boilerplate. However, it re-executes the script top-to-bottom upon user interaction, which requires careful session-state management for heavy workflows.

### Q19. How do you ensure the model doesn't invent non-existent facts during Concept Explanation?
**Answer:** The prompt asks for intuitive analogies and foundational definitions, but anchors technical mechanisms to standard curriculum concepts and highlights common student misconceptions.

### Q20. If you were deploying StudyPilot AI to production, how would you manage user quotas?
**Answer:** We would introduce an API gateway with Redis-based token-bucket rate limiting, tied to authenticated user IDs rather than shared client IPs.

---

# PART 2: INTERMEDIATE LEVEL — DocuMind AI (20 Questions)

### Q21. What is the core limitation of relying purely on an LLM's parametric memory?
**Answer:** Parametric memory is frozen at training cutoff, inaccessible for proprietary or newly published documents, and subject to hallucinations when asked about precise facts. RAG solves this by injecting external, verifiable passages into the prompt context at inference time.

### Q22. Explain the step-by-step pipeline of DocuMind AI.
**Answer:** `Upload ➔ Extract (pypdf/txt) ➔ Clean ➔ Recursive Sliding Window Chunk ➔ Embed (text-embedding-004) ➔ FAISS IndexFlatIP ➔ Query Embedding ➔ Cosine Similarity Search ➔ Grounded LLM Prompting ➔ Source Citation Display`.

### Q23. Why did you choose `faiss.IndexFlatIP` over `IndexFlatL2`?
**Answer:** `IndexFlatIP` computes inner products. When vector embeddings are L2-normalized to unit length ($\|\mathbf{v}\| = 1$), the inner product is mathematically identical to Cosine Similarity. Cosine similarity measures angular orientation rather than vector magnitude, which is the standard metric for text semantic similarity.

### Q24. How do you handle unreadable or scanned PDF documents?
**Answer:** In `DocumentLoader`, we iterate through reader pages extracting selectable text. If the total extracted characters across all pages equals zero, we raise `DocumentLoadingError("PDF appears to be a scanned image or empty. No selectable text could be extracted.")`, informing the user that OCR is required.

### Q25. Why is chunk overlap necessary during text splitting?
**Answer:** If text is partitioned strictly at fixed character boundaries (e.g., every 500 characters), entity names, definitions, or sentences will be sliced in half. Overlapping chunks by 50 characters ensures boundary continuity, preserving semantic context in at least one chunk.

### Q26. How do you prevent hallucinated answers when the uploaded documents do not contain the answer?
**Answer:** In `GroundedQAChain`, the system prompt strictly commands: *"If the context does not contain sufficient facts to answer the question, state: 'Based on the provided documents, there is insufficient evidence to answer this question.' Do not invent facts."* Furthermore, if zero chunks exceed the similarity threshold, generation is bypassed entirely.

### Q27. How does the application preserve document provenance and citations?
**Answer:** During chunking, metadata is bound to every chunk dictionary: `source` (document name), `page_number` (1-indexed page where text appeared), `chunk_id`, and character counts. When chunks are retrieved, these fields are rendered as verifiable source cards.

### Q28. What happens if a user queries an empty vector store?
**Answer:** `FAISSVectorStore.similarity_search` checks `if self.index.ntotal == 0: return []`. The QA chain catches this and immediately returns an "insufficient evidence" response without making any LLM API call.

### Q29. How does document deletion work in an in-memory FAISS index?
**Answer:** `FAISSVectorStore.remove_document(doc_name)` filters out all chunks matching `doc_name`, deletes them from the chunk registry, and re-initializes the FAISS index cleanly to ensure removed documents cannot be retrieved.

### Q30. Why did you use `pypdf` instead of `PyPDF2`?
**Answer:** `PyPDF2` was officially deprecated and merged back into `pypdf` (version 3+). `pypdf` is actively maintained, faster, and provides superior text extraction and bug fixes for modern PDF specifications.

### Q31. What is the mathematical meaning of an embedding vector?
**Answer:** An embedding vector is a point in a high-dimensional continuous vector space (e.g., 768 dimensions). The model maps semantically related words, sentences, or paragraphs to proximal coordinates, allowing semantic similarity to be calculated via vector dot products.

### Q32. How do you distinguish between live embeddings and mock embeddings?
**Answer:** `EmbeddingService` checks API key availability and the `use_mock` flag. Live mode calls Gemini's `embed_content` API returning 768-dim vectors. Mock mode uses a deterministic hash-based algorithm generating 384-dim unit-normalized dense vectors, allowing offline tests to evaluate cosine similarity search without internet access.

### Q33. What is the "Lost in the Middle" problem in LLM retrieval?
**Answer:** Research shows LLMs attend heavily to the beginning and end of long input prompts while frequently ignoring or misinterpreting information situated in the middle. Keeping retrieved chunks concise (Top-3 to Top-5) prevents context dilution and degradation.

### Q34. How would you handle multi-page PDF documents where page numbers must be accurate?
**Answer:** `DocumentLoader` extracts text page-by-page using reader indices, attaching `page_number = idx + 1` to each extracted page dictionary. The chunker propagates this `page_number` down to every chunk derived from that page.

### Q35. What is the time complexity of searching an `IndexFlatIP` index?
**Answer:** For $N$ indexed vectors each of dimension $D$, exhaustive flat search performs $O(N \cdot D)$ floating-point operations. For thousands of chunks, this executes in sub-millisecond time on modern CPUs.

### Q36. When would you switch from `IndexFlatIP` to an approximate index like `IndexIVFFlat` or `IndexHNSW`?
**Answer:** When the knowledge base scales beyond 100,000–1,000,000 vectors where linear scanning introduces noticeable latency. Inverted File (IVF) or Hierarchical Navigable Small World (HNSW) graphs trade a tiny fraction of recall accuracy for logarithmic search speed.

### Q37. What happens if a user uploads two documents with the exact same filename?
**Answer:** In the intermediate application, chunks are tagged with the filename. If identical names are used, chunks merge under that label. In the Advanced application, this is solved by assigning unique SHA256 content hashes to each file.

### Q38. Why is it bad practice to store API keys in Streamlit session state?
**Answer:** Session state can be serialized or inspected during debugging and does not persist across restarts. Environment variables and `.env` files provide secure, isolated configuration.

### Q39. How do you prevent cross-session contamination when multiple users use the app?
**Answer:** In Streamlit, `st.session_state` is completely isolated per active browser tab/session. One user's uploaded documents exist strictly in their own memory space and are never shared with other connected clients.

### Q40. How did you write automated tests for vector retrieval without paid APIs?
**Answer:** In `test_intermediate.py`, we wired `FAISSVectorStore` to `EmbeddingService(use_mock=True)`. The test inserts synthetic documents, queries them, and verifies that cosine similarity scores, top-k limits, and metadata tags are returned correctly.

---

# PART 3: ADVANCED LEVEL — KnowledgeForge AI (20 Questions)

### Q41. Why did you choose FastAPI over Flask or Django for the production backend?
**Answer:** FastAPI offers native asynchronous concurrency (`async`/`await`), automatic OpenAPI/Swagger documentation generation at `/docs`, and native integration with Pydantic v2 for high-speed, type-safe request/response parsing and validation.

### Q42. Explain the LangGraph StateGraph design in KnowledgeForge AI.
**Answer:** The workflow is modeled as an explicit state machine:
`Validate Query ➔ Rewrite Query ➔ Retrieve Chunks ➔ (Conditional Edge) ➔ Hybrid Rerank ➔ Grounded Answer Generation ➔ Grounding Verification`.
If retrieval yields 0 chunks, a conditional branch immediately routes to `insufficient_evidence`, terminating without wasting generation tokens.

### Q43. What is the advantage of using LangGraph over a linear LangChain `SequentialChain`?
**Answer:** Linear chains cannot handle conditional branching, cyclical retries, state inspection, or dynamic fallback nodes cleanly. LangGraph models workflows as explicit, inspectable directed graphs with strongly typed state (`RAGState`), loop-bounds, and deterministic transition logic.

### Q44. How does KnowledgeForge AI achieve document-scoped retrieval?
**Answer:** `DocumentScopedFAISSIndex` accepts an optional `allowed_document_ids` filter. During retrieval, candidate vectors are evaluated, and any chunk whose `document_id` is not present in the allowed list is discarded before reranking and context compilation.

### Q45. How does the Hybrid Reranker combine semantic and lexical scoring?
**Answer:** Dense embeddings can overlook specific keyword matches. The hybrid reranker calculates:
$\text{Score} = (0.65 \cdot \text{Cosine Similarity}) + (0.35 \cdot \text{Token Keyword Overlap}) + \text{Exact Phrase Boost}$.
This ensures passages with high semantic meaning AND explicit query terms are ranked above vaguely related chunks.

### Q46. How does the reranker handle passage deduplication?
**Answer:** After scoring, it computes pairwise Jaccard token overlap between candidate chunks. If a candidate shares $>85\%$ vocabulary overlap with an already selected chunk, it is pruned to avoid redundant token bloat in the prompt.

### Q47. How are document IDs generated in the multi-format ingestion pipeline?
**Answer:** We compute the SHA256 cryptographic hash of the raw file byte stream: `hashlib.sha256(file_bytes).hexdigest()[:16]`. This guarantees stable, reproducible, and tamper-evident IDs: identical files always yield the identical `document_id`.

### Q48. What formats are supported by `MultiFormatIngestor` and how are they parsed?
**Answer:** PDF (`pypdf`), TXT (UTF-8/Latin-1 fallback), and Markdown (`.md`). Markdown and TXT files undergo control-character sanitization and newline normalization before recursive sliding-window chunking.

### Q49. What HTTP status code is returned when a user submits an invalid query?
**Answer:** HTTP 422 Unprocessable Entity (managed automatically by FastAPI and Pydantic validation when `len(query) < 2` or payload structure is malformed).

### Q50. How does the backend defend against prompt injection inside uploaded documents?
**Answer:** Retrieved chunks are isolated within formatted evidence blocks and explicitly labeled as external reference context. System instructions command the model to ignore any embedded directives inside the text and strictly answer the user's research query.

### Q51. Explain how the Grounding Check node works in the LangGraph workflow.
**Answer:** After generating the answer, `check_grounding` inspects the output for citation markers (`[Source X]`) and evaluates whether the text claims insufficient evidence. It returns a calibrated confidence score (e.g. 0.95 with citations) and explanation in the response diagnostics.

### Q52. Why does the API return a `diagnostics` object in `QueryResponse`?
**Answer:** Observability is critical in production RAG systems. The `diagnostics` object provides visibility into raw chunk counts, post-reranking counts, grounding confidence scores, and rewritten query strings for auditing and debugging.

### Q53. How is CORS configured on the FastAPI application?
**Answer:** We apply `CORSMiddleware` with explicit allowed origins, allowing the Streamlit frontend running on port 8501 to communicate with the FastAPI backend running on port 8000 without browser cross-origin blocks.

### Q54. How do you prevent infinite loops in agentic workflows?
**Answer:** Every LangGraph graph has explicit loop-bounds: conditional edges branch strictly towards terminal nodes (`END`) or increment a `retry_count`. If `retry_count >= 1`, the graph transitions immediately to fallback delivery.

### Q55. Describe the structure of the multi-container Docker deployment.
**Answer:** `docker-compose.yml` orchestrates two services: (1) `backend` running FastAPI via Uvicorn on port 8000 with a Docker healthcheck verifying `/health`, and (2) `frontend` running Streamlit on port 8501, configured with `depends_on: { backend: { condition: service_healthy } }`.

### Q56. How is persistent storage handled in the containerized deployment?
**Answer:** A named Docker volume (`kf_storage`) is mounted to `/app/advanced/backend/storage`, ensuring that FAISS indices and document registries survive container restarts.

### Q57. How do you measure retrieval quality in `test_evaluation.py`?
**Answer:** We evaluate **Recall@K** over a synthetic benchmark dataset containing representative engineering questions and expected source documents:
$$\text{Recall@K} = \frac{\text{Relevant Documents Retrieved in Top K}}{\text{Total Known Relevant Documents}}$$
The test asserts that the pipeline satisfies a minimum recall threshold ($\ge 66\%$).

### Q58. What is the difference between synchronous and asynchronous endpoints in FastAPI?
**Answer:** Synchronous endpoints (`def`) run on an external threadpool, while asynchronous endpoints (`async def`) run directly on the event loop. Non-blocking I/O operations (such as HTTP calls or file reads) leverage `async`/`await` for optimal concurrency.

### Q59. What are the key security practices implemented across the backend?
**Answer:** (1) File size caps (15 MB), (2) Extension whitelisting (.pdf, .txt, .md), (3) Sanitization of unprintable bytes, (4) Zero credential hardcoding, (5) Masking of internal filesystem paths in API error responses.

### Q60. If query volume increased 100x, how would you evolve this architecture?
**Answer:** (1) Replace in-memory FAISS with a distributed vector database (e.g. Qdrant or Milvus), (2) Deploy FastAPI on a Kubernetes cluster with Horizontal Pod Autoscalers, (3) Implement Redis caching for frequent vector embeddings and query responses, and (4) Offload heavy ingestion tasks to Celery/RabbitMQ background workers.
