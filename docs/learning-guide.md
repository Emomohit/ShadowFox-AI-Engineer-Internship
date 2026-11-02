# Student Learning & Mentorship Guide
## Demystifying AI Engineering from Fundamentals to Production

---

## 1. Recommended Progressive Learning Order

As a CS/IT student preparing for internship reviews and technical interviews, approach this portfolio in three distinct stages:

```text
Step 1: Master Prompt Engineering & Service Decoupling (Beginner - StudyPilot AI)
   │
   ▼
Step 2: Understand Vector Math, Text Chunking & RAG Retrieval (Intermediate - DocuMind AI)
   │
   ▼
Step 3: Build Typed Microservices, Hybrid Rerankers & LangGraph State Machines (Advanced - KnowledgeForge AI)
```

---

## 2. Beginner Level Walkthrough: StudyPilot AI

### Why Not Just Use ChatGPT?
A naive AI project makes arbitrary API calls directly from a UI script with hardcoded prompts.
In **StudyPilot AI**, we decoupled the application into three independent architectural tiers:
1. **Presentation (`app.py`):** Collects inputs and displays outputs without knowing *how* prompts or LLM drivers work.
2. **Domain Service (`services/study_service.py`):** Enforces input safety (minimum length, character caps, type checking) before any tokens are consumed.
3. **Prompt Catalog (`prompts/study_prompts.py`):** Isolates prompt engineering from business logic. If you need to tweak the MCQ difficulty formatting, you edit the template without touching service logic or UI code.

### Important Code Spotlight: Prompt Injection Delimiters
In `prompts/study_prompts.py`:
```python
"""
STUDENT NOTES:
\"\"\"
{text}
\"\"\"
"""
```
By enclosing user text within triple-quote delimiters (`\"\"\"`), we provide the LLM with a clear boundary separating system instructions from user-provided text. This acts as a first line of defense against prompt injection attempts.

---

## 3. Intermediate Level Walkthrough: DocuMind AI

### What Truly Happens in a RAG Pipeline?
Many beginners believe RAG simply involves pasting a PDF into an LLM prompt. However:
1. LLM context windows have token limits and cost money.
2. The model suffers from the "Lost in the Middle" phenomenon when fed 50-page documents.

**The True RAG Sequence:**
1. **Extraction:** Read PDF streams (`pypdf`) and extract clean, selectable text per page.
2. **Chunking (`chunker.py`):** We use a **sliding window** with `chunk_size=500` and `chunk_overlap=50`. The 50-character overlap ensures that if a sentence is sliced at character 500, the remaining words appear at the start of the next chunk.
3. **Dense Vector Embeddings (`embeddings.py`):** An embedding model transforms a string into a list of numbers (e.g. 768 floats). Words with related semantic concepts point in similar geometric directions.
4. **Vector Search (`vector_store.py`):** Instead of searching for keywords ("database"), vector search understands conceptual queries ("relational storage engine").
5. **Grounded Synthesis (`qa_chain.py`):** We inject the top retrieved chunks into the prompt and instruct the model: *"If the answer is not in this context, say 'insufficient evidence'."*

---

## 4. Advanced Level Walkthrough: KnowledgeForge AI

### Why Do We Need LangGraph?
Linear scripts break when real-world conditions vary. In production, we need a **finite state machine**:
- What if 0 chunks match the query?
  - *Naive script:* Calls the LLM anyway, spends money, and risks hallucinations.
  - *LangGraph:* Conditional edge detects `len(chunks) == 0` and routes directly to the `insufficient_evidence` node.
- What if the query contains ambiguous abbreviations?
  - *LangGraph:* Traverses `rewrite_query` first to expand terms before querying FAISS.

### What is Hybrid Reranking?
Dense embeddings sometimes score a passage higher because it shares general "academic vibes", even if it misses the specific keyword in the query.
In `rag/reranker.py`, we implement a **Hybrid Lexical-Semantic Reranker**:
```python
composite_score = (0.65 * semantic_cosine) + (0.35 * keyword_coverage) + phrase_boost
```
This guarantees that candidates containing both deep semantic relevance AND the user's exact keywords surface at the very top.

### Production Engineering Trade-offs
| Architectural Choice | Advantages | Trade-off / Disadvantage |
| :--- | :--- | :--- |
| **In-Memory FAISS Index** | Sub-millisecond latency, zero external database setup, 100% local. | Must re-index if memory is cleared (mitigated by disk serialization). |
| **Pydantic Validation** | Immediate 422 HTTP errors on invalid payloads; strong runtime typing. | Slight CPU parsing overhead per request. |
| **LangGraph Orchestration** | Clean node separation, inspectable state, conditional branching. | Higher architectural complexity than a single sequential function. |
