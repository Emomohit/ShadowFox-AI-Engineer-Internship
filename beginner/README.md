# StudyPilot AI — AI-Powered Student Study Assistant
> **ShadowFox AI Engineering Internship — Level 1 (Beginner Track)**

---

## 1. Problem Statement
Students frequently struggle with information overload when digesting lengthy academic textbooks, lecture slides, and notes. Traditional study methods often involve passive reading, which leads to weak retention and poor exam readiness. Conversely, generic conversational chatbots lack structured pedagogical frameworks, generate unstructured walls of text, and frequently hallucinate answers or fail to provide structured active-recall assessments.

**StudyPilot AI** bridges this gap by providing an engineering-grade, prompt-engineered student assistant with 4 dedicated, structured workflows:
1. **Smart Notes Summarizer:** Extracts core concepts, hierarchical insights, and revision flashcards.
2. **Quiz Generator:** Formulates calibrated multiple-choice questions (MCQs) with comprehensive answer keys and distractor explanations.
3. **Concept Explainer:** Demystifies intricate technical subjects using plain English, real-world analogies, and misconception warnings.
4. **Answer Improver:** Evaluates draft student answers, elevating clarity, structure, and academic precision with comparative feedback.

---

## 2. Features and Scope
- **Multi-Mode Summarization:** Choose between *Detailed Summary*, *Concise Executive Summary*, and *Flashcard-Style Revision Bullets*.
- **Configurable Quiz Engine:** Select 1 to 10 questions with three difficulty calibrations (*Beginner*, *Medium*, *Hard*).
- **Targeted Conceptual Depth:** Calibrate explanations from *ELI5 / Beginner* to *Undergraduate* and *Senior Engineer*.
- **Comparative Answer Polish:** Side-by-side analysis highlighting vocabulary upgrades, structural flow, and missing technical nuances.
- **Robust Input Validation:** Rejects empty queries, whitespace pads, sub-threshold entries (<10 chars), and protects LLM token budgets (>15,000 chars).
- **Graceful Error Recovery:** Catch and report quota depletion (HTTP 429), malformed keys, and safety blocks cleanly.
- **Dual Runtime Support:** Seamlessly toggle between **Live Gemini API** and an **Offline / Deterministic Test Mode**.

---

## 3. Technology Stack & Design Decisions
| Component | Technology | Rationale |
| :--- | :--- | :--- |
| **Language** | Python 3.12 | Standard runtime for AI/ML engineering, strong typing, wide ecosystem. |
| **Frontend UI** | Streamlit 1.35+ | Rapid, interactive data/AI dashboarding with native state management and tab controls. |
| **LLM SDK** | `google-genai` (Google AI Studio) | High performance, free-tier accessibility, long context windows (Gemini 1.5/2.0 Flash). |
| **Configuration** | `python-dotenv` | Secure credential isolation preventing accidental secret commits. |
| **Testing** | `pytest` | Industry-standard test framework for automated unit and integration verification. |

---

## 4. Architecture Overview
StudyPilot AI adheres to clean separation of concerns:
```text
┌────────────────────────────────────────────────────────┐
│              Streamlit UI Layer (app.py)               │
│   (Inputs, Sliders, Tabs, Load Sample Buttons, Status) │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│         Service Layer (services/study_service.py)       │
│  - Input Validation (Length, Types, Bounds)            │
│  - Prompt Orchestration & Parameter Binding            │
└─────────────┬────────────────────────────┬─────────────┘
              │                            │
              ▼                            ▼
┌───────────────────────────┐  ┌─────────────────────────┐
│ Prompts (study_prompts.py)│  │ LLM Layer (llm_service) │
│ - Summarizer Templates    │  │ - Gemini 1.5 Flash SDK  │
│ - MCQ Exam Templates      │  │ - Deterministic Stub    │
│ - Analogy & Concept Rules │  │ - Timeout / 429 Handler │
└───────────────────────────┘  └─────────────────────────┘
```

---

## 5. End-to-End Workflow
1. **User Action:** Student selects a feature tab, optionally loads a sample input, and configures task parameters (e.g., 5 questions, Medium difficulty).
2. **Client Validation:** `StudyService` sanitizes text, confirms token limits, and asserts constraints.
3. **Prompt Construction:** The sanitized text is injected into specialized prompt templates located in `prompts/study_prompts.py`.
4. **Provider Execution:** `LLMService` dispatches the structured prompt to the Gemini API (or the deterministic local stub in test mode).
5. **Output Parsing & Render:** The response is formatted into readable markdown cards, complete with a one-click download button.

---

## 6. Installation Instructions
Ensure Python 3.10+ is installed. Navigate to the beginner project directory:
```bash
cd beginner
python -m pip install -r requirements.txt
```

---

## 7. Environment Variable Setup
Create a `.env` file in the root repository or inside `beginner/`:
```env
LLM_PROVIDER=gemini
GEMINI_API_KEY=AIzaSy...your_real_key_here
LLM_MODEL=gemini-1.5-flash
```
*Note: If no API key is supplied, StudyPilot AI defaults to Offline Test Mode so you can still test every feature without errors.*

---

## 8. Exact Commands to Run the Application
From the repository root:
```bash
streamlit run beginner/app.py
```
Or from within the `beginner/` directory:
```bash
streamlit run app.py
```
Open your browser to `http://localhost:8501`.

---

## 9. Example Inputs & Expected Outputs
### Notes Summarizer
- **Sample Input:** *"Virtual Memory is a storage allocation scheme in which secondary memory can be addressed as though it were part of main memory. Paging maps virtual to physical frame addresses using a Page Table..."*
- **Expected Output:**
  - `### 🔑 Core Concepts & Definitions`: Virtual Memory, Paging, TLB, Page Fault.
  - `### 📌 Structured Summary`: Clear breakdown of translation flow and performance optimizations.

### Quiz Generator
- **Sample Input:** *"Binary search requires a sorted array and operates in O(log n) time complexity..."*
- **Expected Output:** 3 multiple choice questions with options A, B, C, D, highlighted correct answers, and thorough distractor rationale.

---

## 10. API Usage & Quota Guidelines
- Uses Google Gemini 1.5 Flash (`models/gemini-1.5-flash`).
- Free tier includes 15 Requests Per Minute (RPM) and 1,000,000 Tokens Per Minute (TPM).
- If your account hits quota limits, the UI catches the `RESOURCE_EXHAUSTED` error and alerts you without crashing.

---

## 11. Troubleshooting
- **Error: `Missing valid Gemini API key`**: Verify your `.env` contains `GEMINI_API_KEY=AIzaSy...` or toggle the sidebar switch to "🧪 Demo / Offline Test Mode".
- **Error: `google.genai SDK not installed`**: Run `pip install google-genai`.
- **Streamlit Port Conflict**: Run on an alternate port: `streamlit run beginner/app.py --server.port 8502`.

---

## 12. Limitations & Future Improvements
- **Current Limitations:** Text input is limited to 15,000 characters per request; does not retain multi-turn conversational chat memory across sessions.
- **Future Enhancements:** Direct file upload parsing for DOCX/EPUB, audio flashcard export, and automated spaced repetition scheduling.

---

## 13. Testing Instructions
Run the automated test suite verifying input validation, prompt templates, and error boundaries:
```bash
python -m pytest beginner/tests -v
```

---

## 14. Important Files & File Directory
| File | Responsibility |
| :--- | :--- |
| `app.py` | Streamlit user interface, tabs, form controls, and feedback display. |
| `services/llm_service.py` | Low-level LLM API driver, authentication, error translation, and mock fallback. |
| `services/study_service.py` | Domain service enforcing validation rules and task coordination. |
| `prompts/study_prompts.py` | Decoupled prompt engineering templates. |
| `tests/test_beginner.py` | Pytest verification test suite. |
| `requirements.txt` | Pinned Python dependencies for the beginner level. |
