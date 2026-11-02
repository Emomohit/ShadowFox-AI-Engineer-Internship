# ShadowFox Internship Submission Checklist & Guide
## Pre-Flight Verification & Video Demonstration Checklist

Follow this systematic checklist before submitting your repository and video link to ShadowFox.

---

## 1. Repository Cleanliness & Secret Scanning

- [ ] **Verify No Real Secrets Committed:**
  Run a scan to confirm `.env` files or hardcoded API keys are NOT tracked by Git:
  ```bash
  git status --ignored
  ```
  Ensure `.env` appears under ignored files, and only `.env.example` is committed.
- [ ] **Verify Clean Repository Structure:**
  Confirm root directory contains:
  - `beginner/` (StudyPilot AI)
  - `intermediate/` (DocuMind AI)
  - `advanced/` (KnowledgeForge AI)
  - `shared/` (Mock & testing utilities)
  - `docs/` (Architecture, Learning Guide, Interview Questions, Testing Guide, Checklist)
  - `.env.example`, `.gitignore`, `pytest.ini`, `README.md`, `LICENSE`
- [ ] **Verify Python Bytecode & Cache Pruned:**
  Confirm no `__pycache__` or `.pytest_cache` directories are staged.

---

## 2. Automated Test Verification

Execute all tests from the repository root:
```bash
python -m pytest -v
```
- [ ] **Beginner Level:** 14 / 14 tests passing (`beginner/tests/test_beginner.py`)
- [ ] **Intermediate Level:** 13 / 13 tests passing (`intermediate/tests/test_intermediate.py`)
- [ ] **Advanced Level:** 14 / 14 tests passing (`advanced/tests/`)
- [ ] **Total Pass Rate:** 41 / 41 passing tests.

---

## 3. Mandatory 3 to 5 Minute Video Demonstration

As specified on page 7 of the ShadowFox Internship Task List:
> *"Additionally, it is mandatory to submit a 3 to 5 minute self-recorded video explaining your work. The video must be shared through a Google Drive link in the submission form."*

### Suggested 4-Minute Presentation Script:

| Time | Slide / Screen | Talking Points |
| :--- | :--- | :--- |
| **0:00 – 0:30** | Introduction & Portfolio Overview | Introduce yourself (B.Tech CS/IT student, AI Engineering intern). Present the monorepo architecture and explain the 3-level progression from prompt engineering to RAG and LangGraph orchestration. |
| **0:30 – 1:30** | Level 1: StudyPilot AI Demo | Launch `beginner/app.py`. Show the 4 tabs: Summarizer, Quiz Generator, Concept Explainer, and Answer Improver. Explain the decoupled service layer and prompt catalog. |
| **1:30 – 2:30** | Level 2: DocuMind AI Demo | Launch `intermediate/app.py`. Upload the sample document. Demonstrate sliding-window chunking, FAISS vector search, and ask a question showing grounded generation with page citations. |
| **2:30 – 3:30** | Level 3: KnowledgeForge AI Demo | Show the FastAPI backend (`/docs`) and Streamlit frontend (`advanced/frontend/app.py`). Explain the LangGraph state machine, document-scoped filtering, and hybrid reranking. |
| **3:30 – 4:00** | Code Quality & Conclusion | Show the test suite passing (41/41 tests), explain Docker containerization, and conclude. |

### Video Recording & Sharing Best Practices:
1. Use **OBS Studio** or **Loom** for crisp 1080p screen and microphone capture.
2. Ensure terminal fonts and code are zoomed in to 125%–150% for effortless readability.
3. Upload the recorded MP4 to your personal **Google Drive**.
4. ⚠️ **CRITICAL ACCESS SETTING:** In Google Drive, set Link Access to:
   **"Anyone with the link can view"** (not restricted), so the ShadowFox evaluation mentors can view your demo without requesting access permissions!

---

## 4. Final Internship Submission Form Reminder

When your repository and video link are ready:
1. Push your clean code to your public GitHub repository:
   ```bash
   git remote add origin https://github.com/<your-username>/shadowfox-ai-engineering-portfolio.git
   git push -u origin main
   ```
2. Copy your public GitHub repository URL.
3. Copy your public Google Drive video link.
4. Fill out the official ShadowFox Final Submission Form shared in your respective WhatsApp group.
5. If you require task validation or have doubts, you may contact the assigned mentor at `g2doubts@shadowfox.in`.
