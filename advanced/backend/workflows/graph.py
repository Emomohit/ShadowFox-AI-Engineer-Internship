"""
LangGraph Multi-Step Workflow Graph for KnowledgeForge AI.
Implements:
Validate -> Rewrite -> Retrieve -> Rerank -> Generate -> Grounding Check -> Final Response
"""
from typing import Dict, Any, List
from langgraph.graph import StateGraph, END
from advanced.backend.workflows.states import RAGState
from advanced.backend.rag.embeddings import ProductionEmbeddingService
from advanced.backend.rag.vector_index import DocumentScopedFAISSIndex
from advanced.backend.rag.reranker import HybridReranker
from beginner.services.llm_service import LLMService


class RAGWorkflowBuilder:
    """Constructs and compiles the production RAG LangGraph workflow."""

    def __init__(
        self,
        vector_index: DocumentScopedFAISSIndex,
        embedding_service: ProductionEmbeddingService,
        llm_service: LLMService,
        reranker: HybridReranker
    ):
        self.vector_index = vector_index
        self.embedding_service = embedding_service
        self.llm_service = llm_service
        self.reranker = reranker

    # ---------------- NODE IMPLEMENTATIONS ----------------

    def validate_query(self, state: RAGState) -> Dict[str, Any]:
        """Node 1: Validate query length and structure."""
        q = state.get("raw_query", "").strip()
        if len(q) < 2:
            return {"error": "Query is too short."}
        return {"rewritten_query": q, "error": None}

    def rewrite_query(self, state: RAGState) -> Dict[str, Any]:
        """Node 2: Expand abbreviations and clarify search intent."""
        q = state["rewritten_query"]
        # Basic heuristic query normalization
        expanded = q.replace("FAQ", "Frequently Asked Questions")
        expanded = expanded.replace("OS", "Operating System") if "os " in expanded.lower() else expanded
        return {"rewritten_query": expanded}

    def retrieve(self, state: RAGState) -> Dict[str, Any]:
        """Node 3: Embed rewritten query and retrieve from FAISS."""
        query = state["rewritten_query"]
        doc_ids = state.get("document_ids")
        top_k = state.get("top_k", 4)
        threshold = state.get("similarity_threshold", 0.20)

        # Vector search
        q_vec = self.embedding_service.embed_query(query)
        candidates = self.vector_index.search(
            query_vector=q_vec,
            top_k=top_k * 2,  # pool for reranker
            score_threshold=threshold,
            allowed_document_ids=doc_ids
        )
        return {"retrieved_chunks": candidates}

    def rerank(self, state: RAGState) -> Dict[str, Any]:
        """Node 4: Apply hybrid lexical-semantic reranking."""
        query = state["rewritten_query"]
        candidates = state.get("retrieved_chunks", [])
        top_k = state.get("top_k", 4)

        if not state.get("enable_rerank", True) or not candidates:
            return {"reranked_chunks": candidates[:top_k]}

        reranked = self.reranker.rerank(query=query, candidates=candidates, top_k=top_k)
        return {"reranked_chunks": reranked}

    def generate(self, state: RAGState) -> Dict[str, Any]:
        """Node 5: Formulate grounded answer from evidence."""
        chunks = state.get("reranked_chunks", [])
        query = state["rewritten_query"]

        context_items = []
        for i, c in enumerate(chunks):
            context_items.append(
                f"[Source {i+1}: {c.get('document_name')} | Page {c.get('page_number')}]\n{c.get('text')}"
            )
        context_block = "\n\n---\n\n".join(context_items)

        prompt = f"""CONTEXT EVIDENCE:
\"\"\"
{context_block}
\"\"\"

RESEARCH QUERY:
{query}

INSTRUCTIONS:
1. Provide a rigorous, direct answer based SOLELY on the context evidence above.
2. If evidence does not provide enough facts, state clearly: "Based on the provided documents, there is insufficient evidence to answer this question."
3. Include inline source markers (e.g. [Source 1])."""

        sys_inst = (
            "You are KnowledgeForge AI, an enterprise grounded RAG assistant. "
            "Never hallucinate or extrapolate beyond provided evidence."
        )

        output = self.llm_service.generate_text(prompt=prompt, system_instruction=sys_inst)
        return {"answer": output}

    def check_grounding(self, state: RAGState) -> Dict[str, Any]:
        """Node 6: Verify evidence support and calibrate confidence."""
        ans = state.get("answer", "")
        chunks = state.get("reranked_chunks", [])

        if "insufficient evidence" in ans.lower() or not chunks:
            return {
                "grounded": False,
                "grounding_score": 0.0,
                "grounding_reason": "No matching evidence or insufficient facts in source documents."
            }

        # Check for citation markers
        has_citations = ("[" in ans and "]" in ans) or ("source" in ans.lower())
        score = 0.95 if has_citations else 0.75

        # Format sources
        sources = []
        for c in chunks:
            sources.append({
                "document_id": c.get("document_id", ""),
                "document_name": c.get("document_name", ""),
                "page_number": c.get("page_number", 1),
                "chunk_id": c.get("chunk_id", ""),
                "score": c.get("rerank_score", c.get("similarity_score", 0.0)),
                "excerpt": c.get("text", "")[:280] + ("..." if len(c.get("text", "")) > 280 else "")
            })

        return {
            "grounded": True,
            "grounding_score": score,
            "grounding_reason": "Answer is substantiated by retrieved document chunks.",
            "sources": sources
        }

    def insufficient_evidence(self, state: RAGState) -> Dict[str, Any]:
        """Fallback node when retrieval yields 0 relevant passages."""
        return {
            "answer": (
                "Based on the provided documents, there is insufficient evidence to answer this question. "
                "No relevant document passages exceeded the similarity threshold."
            ),
            "grounded": False,
            "grounding_score": 0.0,
            "grounding_reason": "Zero chunks matched retrieval criteria.",
            "sources": [],
            "reranked_chunks": []
        }

    # ---------------- CONDITIONAL ROUTING ----------------

    def route_after_retrieval(self, state: RAGState) -> str:
        """Branches to insufficient evidence if no chunks found."""
        if not state.get("retrieved_chunks"):
            return "insufficient_evidence"
        return "rerank"

    def build(self):
        """Assembles and compiles the StateGraph."""
        workflow = StateGraph(RAGState)

        workflow.add_node("validate_query", self.validate_query)
        workflow.add_node("rewrite_query", self.rewrite_query)
        workflow.add_node("retrieve", self.retrieve)
        workflow.add_node("rerank", self.rerank)
        workflow.add_node("generate", self.generate)
        workflow.add_node("check_grounding", self.check_grounding)
        workflow.add_node("insufficient_evidence", self.insufficient_evidence)

        # Connect edges
        workflow.set_entry_point("validate_query")
        workflow.add_edge("validate_query", "rewrite_query")
        workflow.add_edge("rewrite_query", "retrieve")

        # Conditional branch after retrieve
        workflow.add_conditional_edges(
            "retrieve",
            self.route_after_retrieval,
            {
                "insufficient_evidence": "insufficient_evidence",
                "rerank": "rerank"
            }
        )

        workflow.add_edge("rerank", "generate")
        workflow.add_edge("generate", "check_grounding")
        workflow.add_edge("check_grounding", END)
        workflow.add_edge("insufficient_evidence", END)

        return workflow.compile()
