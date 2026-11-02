"""
Grounded Question Answering Chain for DocuMind AI.
Enforces strict grounding on retrieved evidence and formats verified citations.
"""
from typing import List, Dict, Any, Optional
from beginner.services.llm_service import LLMService, LLMServiceError


class RAGGenerationError(Exception):
    """Raised when answer generation fails."""
    pass


class GroundedQAChain:
    """Orchestrates grounded response synthesis using retrieved context."""

    SYSTEM_INSTRUCTION = (
        "You are an evidence-grounded AI research assistant. Your task is to answer the user's "
        "question STRICTLY and ONLY using the provided retrieved context. "
        "If the context does not contain sufficient facts to answer the question, state: "
        "'Based on the provided documents, there is insufficient evidence to answer this question.' "
        "Do not invent facts, speculate, or draw upon external knowledge outside the provided context. "
        "When referencing facts, cite the source document name and page number."
    )

    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service

    def build_grounded_prompt(self, question: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
        """Constructs an evidence package and prompt."""
        if not retrieved_chunks:
            return ""

        context_blocks = []
        for i, chunk in enumerate(retrieved_chunks):
            source = chunk.get("source", "Unknown Document")
            page = chunk.get("page_number", "N/A")
            text = chunk.get("text", "")
            score = chunk.get("similarity_score", 0.0)
            context_blocks.append(
                f"[Source {i+1}: {source} | Page: {page} | Score: {score}]\n{text}"
            )

        context_str = "\n\n---\n\n".join(context_blocks)

        prompt = f"""EVIDENCE CONTEXT:
\"\"\"
{context_str}
\"\"\"

USER QUESTION:
{question}

GROUNDED ANSWER (Cite sources inline where applicable, e.g. [Source 1]):"""
        return prompt

    def answer_question(
        self,
        question: str,
        retrieved_chunks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Synthesizes an answer grounded exclusively in retrieved chunks.
        """
        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        clean_question = question.strip()

        # Check if chunks are empty
        if not retrieved_chunks:
            return {
                "answer": (
                    "Based on the provided documents, there is insufficient evidence to answer this question. "
                    "No relevant passages were found in the uploaded documents."
                ),
                "sources": [],
                "grounded": False,
                "status": "insufficient_evidence"
            }

        prompt = self.build_grounded_prompt(clean_question, retrieved_chunks)

        try:
            output = self.llm_service.generate_text(
                prompt=prompt,
                system_instruction=self.SYSTEM_INSTRUCTION
            )
        except LLMServiceError as e:
            raise RAGGenerationError(f"Failed to generate grounded answer: {str(e)}")

        # Format sources list
        sources_summary = []
        for chunk in retrieved_chunks:
            sources_summary.append({
                "source": chunk.get("source", "Unknown"),
                "page_number": chunk.get("page_number", 1),
                "chunk_id": chunk.get("chunk_id", ""),
                "score": chunk.get("similarity_score", 0.0),
                "excerpt": chunk.get("text", "")[:250] + ("..." if len(chunk.get("text", "")) > 250 else "")
            })

        insufficient = (
            "insufficient evidence" in output.lower()
            or "not found in the provided" in output.lower()
            or "cannot be answered" in output.lower()
        )

        return {
            "answer": output,
            "sources": sources_summary,
            "grounded": not insufficient,
            "status": "success" if not insufficient else "insufficient_evidence"
        }
