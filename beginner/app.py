"""
StudyPilot AI - AI-Powered Student Study Assistant
ShadowFox AI Engineering Internship - Beginner Level Project

A production-grade student utility application built with Streamlit and Google Gemini.
Features: Smart Notes Summarizer, Quiz Generator, Concept Explainer, and Answer Improver.
"""
import sys
import os
from dotenv import load_dotenv

# Load environment configuration
load_dotenv()

# Add root directory to sys.path so modules can import shared packages
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import streamlit as st
from beginner.services.llm_service import LLMService, LLMServiceError
from beginner.services.study_service import StudyService

# Set page configuration
st.set_page_config(
    page_title="StudyPilot AI | Student Study Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished, modern typography and clean card layouts
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .feature-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1.2rem;
        margin-bottom: 1rem;
    }
    .status-badge-live {
        background-color: #DCFCE7;
        color: #166534;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .status-badge-mock {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)


def initialize_services(api_key: str, model_name: str, use_mock: bool) -> StudyService:
    """Initializes LLM and Study services with error handling."""
    llm_service = LLMService(
        api_key=api_key,
        model_name=model_name,
        use_mock=use_mock
    )
    return StudyService(llm_service=llm_service)


# ---------------- SIDEBAR CONFIGURATION ----------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/graduation-cap.png", width=64)
    st.title("StudyPilot AI")
    st.caption("ShadowFox AI Internship — Level 1")
    st.divider()

    st.subheader("⚙️ System Status")
    
    api_key = os.getenv("GEMINI_API_KEY", "")
    has_key = bool(api_key and api_key.strip() and api_key != "your_gemini_api_key_here")

    if has_key:
        st.markdown("""
        <div style="background: #F0FDF4; border: 1px solid #BBF7D0; border-radius: 8px; padding: 10px 14px; margin-bottom: 12px;">
            <span style="color: #15803D; font-weight: 700; font-size: 0.9rem;">● Google Gemini AI Active</span>
            <p style="color: #4B5563; font-size: 0.78rem; margin: 4px 0 0 0;">Production API credentials verified</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background: #FEF3C7; border: 1px solid #FDE68A; border-radius: 8px; padding: 10px 14px; margin-bottom: 12px;">
            <span style="color: #92400E; font-weight: 700; font-size: 0.9rem;">● Offline Mode Active</span>
            <p style="color: #4B5563; font-size: 0.78rem; margin: 4px 0 0 0;">Deterministic student stubs running</p>
        </div>
        """, unsafe_allow_html=True)

    model_choice = st.selectbox(
        "Model Version",
        options=["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash"],
        index=0,
        help="Select model tier. 1.5-Flash provides high speed; Pro provides deeper reasoning."
    )

    force_mock = st.toggle("🧪 Offline Demo Mode", value=not has_key)

    if force_mock:
        st.info("Operating in Offline Test Mode. Produces deterministic educational responses without API consumption.")

    st.divider()
    st.markdown("""
    **Architecture Highlights:**
    - 🛡️ Decoupled Service Layer
    - 📝 Task-Specific Prompt Templates
    - ⚡ Client-side & Domain Validation
    - 🔄 Structured Error Recovery
    """)


# ---------------- MAIN INTERFACE ----------------
st.markdown('<h1 class="main-title">🎓 StudyPilot AI</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Your intelligent companion for notes summarization, quiz generation, conceptual mastery, and academic writing polish.</p>', unsafe_allow_html=True)

# Tabbed Navigation for 4 Core Student Features
tab_summarize, tab_quiz, tab_explain, tab_improve = st.tabs([
    "📝 Smart Summarizer",
    "🎯 Quiz Generator",
    "💡 Concept Explainer",
    "✨ Answer Improver"
])

# Initialize Service
try:
    study_service = initialize_services(
        api_key=api_key,
        model_name=model_choice,
        use_mock=force_mock
    )
except LLMServiceError as e:
    st.error(f"Configuration Error: {str(e)}")
    st.stop()


# ==============================================================================
# TAB 1: SMART SUMMARIZER
# ==============================================================================
with tab_summarize:
    st.subheader("Transform Raw Study Notes into High-Yield Insights")
    st.write("Extract critical definitions, core mechanisms, and key takeaways from textbooks or lecture transcripts.")

    col_btn, _ = st.columns([1, 4])
    if col_btn.button("📥 Load Sample Operating Systems Notes", key="sample_notes_btn"):
        st.session_state["notes_input"] = (
            "Virtual Memory is a storage allocation scheme in which secondary memory can be addressed "
            "as though it were part of main memory. The addresses a program may use to reference memory are "
            "distinguished from the addresses the memory system uses to identify physical storage sites.\n\n"
            "Key Mechanisms:\n"
            "1. Paging: A memory management scheme by which a computer stores and retrieves data from secondary "
            "storage for use in main memory in same-size blocks called pages.\n"
            "2. Page Table: Maps virtual addresses to physical frame addresses. Each page table entry includes "
            "valid/invalid bits, dirty bits, and access control bits.\n"
            "3. Translation Lookaside Buffer (TLB): A hardware associative cache used to reduce the time taken to "
            "access a user memory location. If a virtual address has a hit in TLB, memory access is instantaneous.\n"
            "4. Page Fault: An interrupt generated when an accessed page is not currently mapped into physical RAM."
        )

    notes_text = st.text_area(
        "Paste your study notes here:",
        value=st.session_state.get("notes_input", ""),
        height=180,
        placeholder="Paste lecture transcript, textbook excerpts, or draft notes here (min 10 characters)..."
    )

    col1, col2 = st.columns([2, 1])
    with col1:
        summary_style = st.radio(
            "Summary Format:",
            options=["detailed", "concise", "revision"],
            format_func=lambda x: {
                "detailed": "Detailed Summary with Deep Insights",
                "concise": "Concise Executive Summary (3-4 sentences)",
                "revision": "High-Yield Revision Flashcard Bullets"
            }[x],
            horizontal=True
        )
    with col2:
        summarize_button = st.button("🚀 Generate Summary", type="primary", use_container_width=True)

    if summarize_button:
        try:
            with st.spinner("Analyzing notes and distilling key concepts..."):
                res = study_service.summarize_notes(notes=notes_text, summary_type=summary_style)
                st.success(f"Summary Generated Successfully ({res['char_count']} chars analyzed)")
                st.markdown("### Output")
                st.markdown(res["summary"])
                st.download_button(
                    "💾 Download Summary Markdown",
                    data=res["summary"],
                    file_name="StudyPilot_Summary.md",
                    mime="text/markdown"
                )
        except (ValueError, LLMServiceError) as err:
            st.error(f"⚠️ {str(err)}")


# ==============================================================================
# TAB 2: QUIZ GENERATOR
# ==============================================================================
with tab_quiz:
    st.subheader("Generate Active Recall Multiple Choice Quizzes")
    st.write("Test your retention with pedagogically calibrated MCQs complete with answer keys and explanations.")

    col_btn_q, _ = st.columns([1, 4])
    if col_btn_q.button("📥 Load Sample Algorithms Notes", key="sample_quiz_btn"):
        st.session_state["quiz_notes_input"] = (
            "Binary Search is an efficient algorithm for finding an item from a sorted list of items. "
            "It works by repeatedly dividing in half the portion of the list that could contain the item, until "
            "the possible locations are narrowed down to just one.\n\n"
            "Time Complexity: Worst and Average case is O(log n). Best case is O(1) when the target is at middle.\n"
            "Space Complexity: O(1) for iterative implementation, O(log n) for recursive call stack.\n"
            "Precondition: The underlying collection MUST be sorted in monotonic order beforehand."
        )

    quiz_notes = st.text_area(
        "Source Notes for Quiz Generation:",
        value=st.session_state.get("quiz_notes_input", ""),
        height=160,
        placeholder="Enter notes to generate quiz questions from..."
    )

    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        num_q = st.slider("Number of Questions", min_value=1, max_value=10, value=3)
    with c2:
        diff = st.selectbox("Difficulty Level", ["Beginner", "Medium", "Hard"], index=1)
    with c3:
        st.write("")
        st.write("")
        generate_quiz_btn = st.button("🎯 Generate Quiz", type="primary", use_container_width=True)

    if generate_quiz_btn:
        try:
            with st.spinner(f"Formulating {num_q} {diff}-level questions with verified explanations..."):
                res = study_service.generate_quiz(notes=quiz_notes, num_questions=num_q, difficulty=diff)
                st.success("Quiz Generated Successfully!")
                st.markdown(res["quiz"])
                st.download_button(
                    "💾 Download Quiz as Markdown",
                    data=res["quiz"],
                    file_name="StudyPilot_Quiz.md",
                    mime="text/markdown"
                )
        except (ValueError, LLMServiceError) as err:
            st.error(f"⚠️ {str(err)}")


# ==============================================================================
# TAB 3: CONCEPT EXPLAINER
# ==============================================================================
with tab_explain:
    st.subheader("Intuitive Concept Explainer")
    st.write("Demystify difficult concepts with real-world analogies, step-by-step logic, and misconception traps.")

    col_btn_e, _ = st.columns([1, 4])
    if col_btn_e.button("📥 Load Example Topic", key="sample_concept_btn"):
        st.session_state["concept_input"] = "How Database Indexing (B-Trees) works"

    concept_query = st.text_input(
        "Enter Concept or Topic to Explain:",
        value=st.session_state.get("concept_input", ""),
        placeholder="e.g. Asymmetric Cryptography, Backpropagation, MapReduce, CAP Theorem..."
    )

    col_aud, col_act = st.columns([2, 1])
    with col_aud:
        audience = st.select_slider(
            "Target Audience Depth:",
            options=["Beginner (ELI5)", "Intermediate (Undergrad)", "Advanced (Senior Engineer)"],
            value="Beginner (ELI5)"
        )
    with col_act:
        st.write("")
        explain_btn = st.button("💡 Explain Concept", type="primary", use_container_width=True)

    if explain_btn:
        try:
            with st.spinner("Constructing intuitive conceptual walkthrough..."):
                res = study_service.explain_concept(concept=concept_query, audience_level=audience)
                st.markdown(res["explanation"])
        except (ValueError, LLMServiceError) as err:
            st.error(f"⚠️ {str(err)}")


# ==============================================================================
# TAB 4: ANSWER IMPROVER
# ==============================================================================
with tab_improve:
    st.subheader("Academic Writing & Technical Answer Improver")
    st.write("Upgrade draft answers into rigorous, high-scoring exam solutions with comparative feedback.")

    col_btn_a, _ = st.columns([1, 4])
    if col_btn_a.button("📥 Load Sample Student Answer", key="sample_answer_btn"):
        st.session_state["answer_q"] = "What is the difference between TCP and UDP?"
        st.session_state["answer_draft"] = (
            "tcp is slower because it checks everything and makes sure packets arrive. udp is faster "
            "because it just sends them without checking. tcp has a handshake and udp doesn't."
        )

    question_prompt = st.text_input(
        "Exam Question / Prompt (Optional):",
        value=st.session_state.get("answer_q", ""),
        placeholder="e.g., Explain the difference between Process and Thread..."
    )

    draft_text = st.text_area(
        "Your Draft Answer:",
        value=st.session_state.get("answer_draft", ""),
        height=140,
        placeholder="Paste your rough answer here..."
    )

    improve_btn = st.button("✨ Polish & Improve Answer", type="primary")

    if improve_btn:
        try:
            with st.spinner("Reviewing grammar, technical vocabulary, and structure..."):
                res = study_service.improve_answer(draft_answer=draft_text, question=question_prompt)
                st.success("Feedback & Polished Version Ready")
                st.markdown(res["improved_answer"])
        except (ValueError, LLMServiceError) as err:
            st.error(f"⚠️ {str(err)}")
