"""
KnowledgeForge AI - Production Streamlit Frontend
ShadowFox AI Engineering Internship - Advanced Level Project
"""
import sys
import os
import requests
from dotenv import load_dotenv

load_dotenv()

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import streamlit as st

# Set page config
st.set_page_config(
    page_title="KnowledgeForge AI | Enterprise RAG Assistant",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .kf-header {
        font-size: 2.3rem;
        font-weight: 800;
        color: #1E293B;
        margin-bottom: 0.1rem;
    }
    .kf-sub {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.2rem;
    }
    .langgraph-badge {
        background: linear-gradient(90deg, #3B82F6, #10B981);
        color: white;
        padding: 4px 12px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .diag-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Backend URL configuration
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# Check backend health
def check_backend_health():
    try:
        r = requests.get(f"{BACKEND_URL}/health", timeout=2)
        if r.status_code == 200:
            return True, r.json()
        return False, None
    except Exception:
        return False, None

is_backend_online, backend_meta = check_backend_health()

# ---------------- SIDEBAR CONTROLS ----------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/shield.png", width=64)
    st.title("KnowledgeForge AI")
    st.caption("ShadowFox AI Internship — Level 3 (Production RAG)")
    st.divider()

    st.subheader("🌐 Backend Status")
    if is_backend_online:
        st.success(f"● Connected to FastAPI ({BACKEND_URL})")
        st.caption(f"Provider: `{backend_meta.get('provider')}` | Docs: `{backend_meta.get('total_documents')}` | Chunks: `{backend_meta.get('total_indexed_chunks')}`")
    else:
        st.error(f"● Backend Offline ({BACKEND_URL})")
        st.caption("Start FastAPI backend using: `uvicorn advanced.backend.main:app --port 8000`")

    st.divider()
    st.subheader("🎛️ Pipeline Parameters")
    top_k = st.slider("Top-K Retrieved Chunks", min_value=1, max_value=8, value=4)
    similarity_thresh = st.slider("Similarity Cutoff", min_value=0.0, max_value=0.8, value=0.20, step=0.05)
    enable_rerank = st.toggle("Enable Hybrid Reranker", value=True)

    st.divider()
    st.subheader("📚 Active Knowledge Base")
    doc_options = {}
    if is_backend_online:
        try:
            r = requests.get(f"{BACKEND_URL}/documents", timeout=3)
            if r.status_code == 200:
                docs_data = r.json().get("documents", [])
                for d in docs_data:
                    doc_options[d["document_id"]] = f"{d['filename']} ({d['chunks_count']} chunks)"
                    col1, col2 = st.columns([3, 1])
                    col1.write(f"📄 {d['filename']}")
                    if col2.button("🗑️", key=f"del_{d['document_id']}"):
                        requests.delete(f"{BACKEND_URL}/documents/{d['document_id']}")
                        st.rerun()
        except Exception:
            pass

    selected_docs = st.multiselect(
        "Scope Search to Specific Documents (Optional):",
        options=list(doc_options.keys()),
        format_func=lambda x: doc_options.get(x, x),
        help="Leave blank to query all indexed documents in the knowledge base."
    )


# ---------------- MAIN UI & TABS ----------------
st.markdown('<h1 class="kf-header">🛡️ KnowledgeForge AI</h1>', unsafe_allow_html=True)
st.markdown(
    '<span class="langgraph-badge">LangGraph Multi-Step Orchestration • FAISS FlatIP • Hybrid Reranking</span>',
    unsafe_allow_html=True
)
st.write("")

tab_chat, tab_ingest, tab_workflow = st.tabs([
    "💬 Grounded Research Q&A",
    "📥 Multi-Format Ingestion",
    "🕸️ LangGraph Architecture"
])


# ==============================================================================
# TAB 1: GROUNDED RESEARCH Q&A
# ==============================================================================
with tab_chat:
    st.subheader("Query the Knowledge Repository")

    col_btn_load, _ = st.columns([1, 4])
    if col_btn_load.button("📥 Load Sample Question", key="sample_kf_q"):
        st.session_state["kf_query"] = "What are the core stages in the LangGraph RAG workflow?"

    query_text = st.text_input(
        "Enter research query:",
        value=st.session_state.get("kf_query", ""),
        placeholder="e.g. How does document-scoped filtering work in the FAISS vector index?"
    )

    if st.button("🚀 Run Workflow Query", type="primary", disabled=not is_backend_online):
        if not query_text.strip():
            st.error("Please enter a valid query.")
        else:
            payload = {
                "query": query_text,
                "document_ids": selected_docs if selected_docs else None,
                "top_k": top_k,
                "similarity_threshold": similarity_thresh,
                "enable_rerank": enable_rerank,
                "session_id": "kf_demo_session"
            }

            try:
                with st.spinner("Traversing LangGraph nodes: Validate -> Rewrite -> Retrieve -> Rerank -> Generate -> Grounding Check..."):
                    res = requests.post(f"{BACKEND_URL}/query", json=payload, timeout=20)
                    if res.status_code == 200:
                        data = res.json()
                        st.markdown("### 🤖 Synthesized Answer")
                        if data["grounded"]:
                            st.markdown(data["answer"])
                        else:
                            st.warning(data["answer"])

                        # Citations
                        st.markdown("### 📚 Source Citations")
                        if data.get("sources"):
                            for s in data["sources"]:
                                st.markdown(f"""
                                <div class="diag-card">
                                    <b>📄 {s['document_name']}</b> — Page {s['page_number']} | <i>Relevance Score: {s['score']}</i><br>
                                    <span style="color: #475569;">"{s['excerpt']}"</span>
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.info("No sources exceeded the similarity cutoff.")

                        # Observability & Diagnostics
                        with st.expander("🔍 Observability & Workflow Diagnostics"):
                            diag = data.get("diagnostics", {})
                            st.json({
                                "Rewritten Query": data.get("rewritten_query"),
                                "Chunks Retrieved": diag.get("retrieved_count"),
                                "Chunks After Reranking": diag.get("reranked_count"),
                                "Grounding Score": diag.get("grounding_score"),
                                "Grounding Reason": diag.get("grounding_reason"),
                                "Session ID": data.get("session_id")
                            })
                    else:
                        st.error(f"Backend returned error ({res.status_code}): {res.text}")
            except Exception as e:
                st.error(f"Query dispatch error: {str(e)}")


# ==============================================================================
# TAB 2: MULTI-FORMAT INGESTION
# ==============================================================================
with tab_ingest:
    st.subheader("Ingest PDF, TXT, or Markdown (MD)")
    st.write("Upload documents to decompose into stable SHA256 chunks and index into FAISS.")

    col_btn_load_file, _ = st.columns([1, 4])
    if col_btn_load_file.button("📥 Load Sample Technical Specification (MD)", key="sample_md_btn"):
        sample_filename = "KnowledgeForge_Spec.md"
        sample_md_bytes = (
            "# KnowledgeForge AI Technical Specification\n\n"
            "## 1. System Architecture\n"
            "KnowledgeForge AI utilizes an agentic LangGraph pipeline coupled with FastAPI and FAISS. "
            "Incoming user queries are systematically rewritten into unambiguous search requests, "
            "then evaluated against dense vector embeddings generated via Gemini text-embedding-004.\n\n"
            "## 2. Hybrid Reranking\n"
            "Retrieved chunks undergo hybrid reranking: combining semantic cosine similarity with BM25 token overlap. "
            "Passages with a token Jaccard similarity >0.85 are automatically deduplicated to prevent context stuffing.\n\n"
            "## 3. Strict Grounding\n"
            "The generation node is constrained to evidence-grounded facts. If zero chunks match the threshold, "
            "the system transitions to the insufficient_evidence fallback node."
        ).encode("utf-8")

        if is_backend_online:
            files = {"file": (sample_filename, sample_md_bytes, "text/markdown")}
            data = {"chunk_size": 400, "chunk_overlap": 40}
            r = requests.post(f"{BACKEND_URL}/documents/upload", files=files, data=data)
            if r.status_code == 201:
                st.success(f"✓ Ingested '{sample_filename}' ({r.json()['chunks_count']} chunks)!")
                st.rerun()

    uploaded_files = st.file_uploader(
        "Upload files (PDF, TXT, MD):",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True
    )

    if uploaded_files and st.button("🚀 Process & Ingest Files", type="primary", disabled=not is_backend_online):
        for f in uploaded_files:
            file_bytes = f.read()
            files = {"file": (f.name, file_bytes, f.type or "application/octet-stream")}
            data = {"chunk_size": 500, "chunk_overlap": 50}
            with st.spinner(f"Ingesting {f.name}..."):
                r = requests.post(f"{BACKEND_URL}/documents/upload", files=files, data=data)
                if r.status_code == 201:
                    st.success(f"✓ Successfully indexed '{f.name}'!")
                else:
                    st.error(f"Failed to ingest '{f.name}': {r.text}")
        st.rerun()


# ==============================================================================
# TAB 3: LANGGRAPH ARCHITECTURE
# ==============================================================================
with tab_workflow:
    st.subheader("LangGraph State Machine Architecture")
    st.markdown("""
    ```mermaid
    graph TD
        A([Entry: User Query]) --> B[Validate Query]
        B --> C[Rewrite & Clarify Query]
        C --> D[Retrieve Chunks from FAISS]
        D -->|Chunks Found > 0| E[Hybrid Reranker & Deduplication]
        D -->|Chunks == 0| F[Insufficient Evidence Node]
        E --> G[Grounded Answer Generation]
        G --> H[Check Grounding & Citations]
        H --> I([End: Return Typed QueryResponse])
        F --> I
    ```
    """)
    st.markdown("""
    **Core Production Principles:**
    1. **Deterministic Branching:** Never wastes LLM tokens if 0 chunks meet similarity thresholds.
    2. **State Transparency:** Every node reads from and mutates typed `RAGState`.
    3. **Deduplication:** Prevents redundant source chunks from overwhelming the context window.
    4. **Defensive Prompting:** System instructions treat document chunks as untrusted evidence.
    """)
