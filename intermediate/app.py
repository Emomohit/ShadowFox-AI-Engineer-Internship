"""
DocuMind AI - Document-Based Question Answering Assistant
ShadowFox AI Engineering Internship - Intermediate Level Project

A complete Retrieval-Augmented Generation (RAG) pipeline built with:
Document Extraction -> Recursive Chunking -> Embeddings -> FAISS Vector Index -> Grounded LLM Generation
"""
import sys
import os
from dotenv import load_dotenv

# Load environment configuration
load_dotenv()

# Add root directory to sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import streamlit as st
from intermediate.ingestion.loader import DocumentLoader, DocumentLoadingError
from intermediate.ingestion.chunker import TextChunker
from intermediate.retrieval.embeddings import EmbeddingService, EmbeddingError
from intermediate.retrieval.vector_store import FAISSVectorStore
from intermediate.generation.qa_chain import GroundedQAChain, RAGGenerationError
from beginner.services.llm_service import LLMService, LLMServiceError

# Streamlit Page Setup
st.set_page_config(
    page_title="DocuMind AI | Document Q&A Assistant",
    page_icon="📑",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 0.2rem;
    }
    .pipeline-badge {
        background-color: #EEF2F6;
        color: #334155;
        font-family: monospace;
        font-size: 0.85rem;
        padding: 4px 8px;
        border-radius: 4px;
        margin-bottom: 1rem;
        display: inline-block;
    }
    .citation-box {
        background-color: #F8FAFC;
        border-left: 4px solid #3B82F6;
        padding: 10px 14px;
        border-radius: 0 8px 8px 0;
        margin-bottom: 8px;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)


# Session State Initialization
if "vector_store" not in st.session_state:
    st.session_state.vector_store = FAISSVectorStore()
if "uploaded_doc_metadata" not in st.session_state:
    st.session_state.uploaded_doc_metadata = {}
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ---------------- SIDEBAR CONTROLS ----------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/search-in-list.png", width=64)
    st.title("DocuMind AI")
    st.caption("ShadowFox AI Internship — Level 2 (RAG)")
    st.divider()

    st.subheader("⚙️ System Status")
    api_key = os.getenv("GEMINI_API_KEY", "")
    has_key = bool(api_key and api_key.strip() and api_key != "your_gemini_api_key_here")

    if has_key:
        st.markdown("""
        <div style="background: #F0FDF4; border: 1px solid #BBF7D0; border-radius: 8px; padding: 10px 14px; margin-bottom: 12px;">
            <span style="color: #15803D; font-weight: 700; font-size: 0.9rem;">● Gemini & FAISS Active</span>
            <p style="color: #4B5563; font-size: 0.78rem; margin: 4px 0 0 0;">text-embedding-004 + Gemini 1.5</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background: #FEF3C7; border: 1px solid #FDE68A; border-radius: 8px; padding: 10px 14px; margin-bottom: 12px;">
            <span style="color: #92400E; font-weight: 700; font-size: 0.9rem;">● Offline Vector Mode</span>
            <p style="color: #4B5563; font-size: 0.78rem; margin: 4px 0 0 0;">Deterministic local embedding stubs</p>
        </div>
        """, unsafe_allow_html=True)

    force_mock = st.toggle("🧪 Offline Demo Mode", value=not has_key)

    if force_mock:
        st.caption("Using deterministic dense hash embeddings & local LLM stubs.")

    st.divider()
    st.subheader("🎛️ RAG Hyperparameters")
    chunk_size = st.slider("Chunk Size (characters)", min_value=150, max_value=1200, value=500, step=50)
    chunk_overlap = st.slider("Chunk Overlap (characters)", min_value=0, max_value=200, value=50, step=10)
    top_k = st.slider("Top-K Retrieved Chunks", min_value=1, max_value=8, value=3)
    similarity_threshold = st.slider("Min Similarity Threshold", min_value=0.0, max_value=0.8, value=0.1, step=0.05)

    st.divider()
    st.subheader("📂 Active Documents")
    active_docs = st.session_state.vector_store.get_indexed_documents()
    if active_docs:
        for doc in active_docs:
            col_d1, col_d2 = st.columns([3, 1])
            col_d1.write(f"📄 **{doc}**")
            if col_d2.button("🗑️", key=f"del_{doc}", help="Remove from index"):
                st.session_state.vector_store.remove_document(doc)
                st.session_state.uploaded_doc_metadata.pop(doc, None)
                st.rerun()

        if st.button("Clear Entire Index", type="secondary", use_container_width=True):
            st.session_state.vector_store.clear()
            st.session_state.uploaded_doc_metadata.clear()
            st.session_state.chat_history.clear()
            st.success("Vector store reset!")
            st.rerun()
    else:
        st.info("No documents currently indexed.")


# Initialize Services
try:
    embedding_service = EmbeddingService(
        api_key=api_key,
        use_mock=force_mock
    )
    llm_service = LLMService(
        api_key=api_key,
        use_mock=force_mock
    )
    qa_chain = GroundedQAChain(llm_service=llm_service)
    chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
except (EmbeddingError, LLMServiceError) as err:
    st.error(f"Initialization Error: {str(err)}")
    st.stop()


# ---------------- MAIN HEADER & TABS ----------------
st.markdown('<h1 class="main-header">📑 DocuMind AI</h1>', unsafe_allow_html=True)
st.markdown(
    '<span class="pipeline-badge">Pipeline: Ingest ➔ Extract ➔ Clean ➔ Recursive Chunk ➔ FAISS Index ➔ Top-K Retrieval ➔ Grounded Answer</span>',
    unsafe_allow_html=True
)

tab_upload, tab_qa, tab_inspect = st.tabs([
    "📤 Document Upload & Ingestion",
    "💬 Grounded Question Answering",
    "🔍 Knowledge Base Inspector"
])


# ==============================================================================
# TAB 1: DOCUMENT UPLOAD & INGESTION
# ==============================================================================
with tab_upload:
    st.subheader("Ingest Knowledge Documents (PDF & TXT)")
    st.write("Upload course slides, research papers, or syllabus text to index into your personal FAISS vector store.")

    col_btn_load, _ = st.columns([1, 3])
    if col_btn_load.button("📥 Load Sample Architecture Overview (TXT)", key="sample_ingest_btn"):
        sample_filename = "ShadowFox_RAG_Architecture.txt"
        sample_content = (
            "SHADOWFOX AI ARCHITECTURE SPECIFICATION\n\n"
            "Section 1: Ingestion Pipeline\n"
            "The document ingestion pipeline processes unstructured files (PDF, TXT) by stripping non-printable "
            "characters, isolating page boundaries, and segmenting content into semantically coherent chunks. "
            "Chunk size is calibrated between 400 and 800 characters with a 10% overlap to prevent loss of boundary context.\n\n"
            "Section 2: Embedding Generation & FAISS Indexing\n"
            "Chunks are converted into high-dimensional vector representations using Gemini text-embedding-004. "
            "Vectors are L2-normalized and indexed in FAISS using an IndexFlatIP structure, allowing exact cosine similarity "
            "computation at sub-millisecond latency.\n\n"
            "Section 3: Grounded Answer Synthesis\n"
            "When a query arrives, it is embedded into the vector space, and the top-k nearest chunk neighbors are retrieved. "
            "The generation prompt strictly enforces that if the answer cannot be established directly from retrieved chunks, "
            "the system explicitly reports insufficient evidence rather than hallucinating facts."
        ).encode("utf-8")

        with st.spinner("Processing and indexing sample document..."):
            pages = DocumentLoader.load_file(sample_content, sample_filename)
            chunks = chunker.split_pages(pages)
            texts = [c["text"] for c in chunks]
            embeddings = embedding_service.embed_documents(texts)
            st.session_state.vector_store.add_documents(chunks, embeddings)
            st.session_state.uploaded_doc_metadata[sample_filename] = {
                "pages": len(pages),
                "chunks": len(chunks)
            }
            st.success(f"Successfully indexed '{sample_filename}' ({len(chunks)} chunks in FAISS index)!")

    uploaded_files = st.file_uploader(
        "Upload PDF or TXT files:",
        type=["pdf", "txt"],
        accept_multiple_files=True
    )

    if uploaded_files:
        if st.button("🚀 Process & Index Uploaded Documents", type="primary"):
            progress_bar = st.progress(0)
            total_files = len(uploaded_files)

            for i, uploaded_file in enumerate(uploaded_files):
                fname = uploaded_file.name
                file_bytes = uploaded_file.read()

                try:
                    with st.spinner(f"Extracting & chunking '{fname}'..."):
                        pages = DocumentLoader.load_file(file_bytes, fname)
                        chunks = chunker.split_pages(pages)

                        st.write(f"⚙️ Embedding {len(chunks)} chunks for '{fname}'...")
                        chunk_texts = [c["text"] for c in chunks]
                        embeddings = embedding_service.embed_documents(chunk_texts)

                        st.session_state.vector_store.add_documents(chunks, embeddings)
                        st.session_state.uploaded_doc_metadata[fname] = {
                            "pages": len(pages),
                            "chunks": len(chunks)
                        }
                        st.success(f"✓ Ingested '{fname}' ({len(pages)} pages, {len(chunks)} chunks)")

                except (DocumentLoadingError, EmbeddingError, ValueError) as e:
                    st.error(f"❌ Failed to ingest '{fname}': {str(e)}")

                progress_bar.progress((i + 1) / total_files)

            st.rerun()


# ==============================================================================
# TAB 2: GROUNDED QUESTION ANSWERING
# ==============================================================================
with tab_qa:
    st.subheader("Ask Grounded Questions Against Indexed Documents")

    if st.session_state.vector_store.total_chunks == 0:
        st.warning("⚠️ No documents have been indexed yet. Please upload a document in the 'Document Upload' tab first.")
    else:
        st.write(f"Index Status: **{st.session_state.vector_store.total_chunks} active chunks** across {len(st.session_state.uploaded_doc_metadata)} documents.")

        col_q1, col_q2 = st.columns([1, 4])
        if col_q1.button("💡 Load Sample Query", key="sample_query_btn"):
            st.session_state["query_input"] = "How does the ingestion pipeline handle chunk boundaries and overlap?"

        user_query = st.text_input(
            "Enter your question:",
            value=st.session_state.get("query_input", ""),
            placeholder="e.g. What chunk size is recommended? What happens when evidence is insufficient?"
        )

        if st.button("🔍 Search & Answer", type="primary"):
            if not user_query.strip():
                st.error("Please enter a question.")
            else:
                try:
                    with st.spinner("Embedding query & searching FAISS index..."):
                        # 1. Embed query
                        q_vec = embedding_service.embed_text(user_query)

                        # 2. Vector search
                        retrieved_chunks = st.session_state.vector_store.similarity_search(
                            query_vector=q_vec,
                            top_k=top_k,
                            score_threshold=similarity_threshold
                        )

                    with st.spinner("Synthesizing grounded response..."):
                        # 3. Grounded Generation
                        rag_response = qa_chain.answer_question(
                            question=user_query,
                            retrieved_chunks=retrieved_chunks
                        )

                    # Display Answer
                    st.markdown("### 🤖 Answer")
                    if rag_response["grounded"]:
                        st.markdown(rag_response["answer"])
                    else:
                        st.warning(f"⚠️ {rag_response['answer']}")

                    # Display Supporting Sources
                    st.markdown("### 📚 Supporting Evidence & Citations")
                    if rag_response["sources"]:
                        for s in rag_response["sources"]:
                            st.markdown(f"""
                            <div class="citation-box">
                                <b>📄 {s['source']}</b> — Page {s['page_number']} | <i>Similarity Score: {s['score']}</i><br>
                                <span style="color: #475569;">"{s['excerpt']}"</span>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.info("No source passages matched the retrieval threshold.")

                except (EmbeddingError, RAGGenerationError, ValueError) as err:
                    st.error(f"Search Error: {str(err)}")


# ==============================================================================
# TAB 3: KNOWLEDGE BASE INSPECTOR
# ==============================================================================
with tab_inspect:
    st.subheader("Inspect Indexed Chunks and Vector Embeddings")
    if not st.session_state.vector_store.chunks:
        st.info("The vector store is currently empty.")
    else:
        st.write(f"Total Chunks: **{len(st.session_state.vector_store.chunks)}**")
        for idx, chunk in enumerate(st.session_state.vector_store.chunks):
            with st.expander(f"Chunk #{idx+1}: {chunk['chunk_id']} ({chunk['char_count']} chars)"):
                st.write(f"**Document:** `{chunk['source']}` | **Page:** `{chunk['page_number']}`")
                st.code(chunk["text"], language="text")
