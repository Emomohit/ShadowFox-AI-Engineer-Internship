"""
Study Service Layer for StudyPilot AI.
Implements the core business logic, input validation, and task orchestration
for Note Summarization, Quiz Generation, Concept Explanation, and Answer Improvement.
"""
from typing import Dict, Any, Optional
from beginner.prompts.study_prompts import StudyPrompts
from beginner.services.llm_service import LLMService, LLMServiceError


class StudyService:
    """Orchestrates study assistant workflows and enforces domain constraints."""

    MAX_INPUT_CHARS = 15000
    MIN_INPUT_CHARS = 10

    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service

    def _validate_text(self, text: str, field_name: str = "Input text") -> str:
        """Sanitizes and validates user input text."""
        if not text or not text.strip():
            raise ValueError(f"{field_name} cannot be empty. Please provide some content.")
        
        cleaned = text.strip()
        if len(cleaned) < self.MIN_INPUT_CHARS:
            raise ValueError(f"{field_name} is too short ({len(cleaned)} chars). Minimum required is {self.MIN_INPUT_CHARS} characters.")
        
        if len(cleaned) > self.MAX_INPUT_CHARS:
            raise ValueError(
                f"{field_name} exceeds the maximum limit of {self.MAX_INPUT_CHARS} characters "
                f"(current: {len(cleaned)} chars). Please condense your input."
            )
        return cleaned

    def summarize_notes(self, notes: str, summary_type: str = "detailed") -> Dict[str, Any]:
        """Validates and summarizes student notes."""
        valid_notes = self._validate_text(notes, field_name="Study Notes")
        prompt = StudyPrompts.get_summarizer_prompt(valid_notes, summary_type=summary_type)
        output = self.llm_service.generate_text(
            prompt=prompt,
            system_instruction="You are an expert academic tutor. Summarize clearly, preserving key insights and technical rigor."
        )
        return {
            "summary": output,
            "char_count": len(valid_notes),
            "summary_type": summary_type,
            "status": "success"
        }

    def generate_quiz(self, notes: str, num_questions: int = 5, difficulty: str = "Medium") -> Dict[str, Any]:
        """Generates a multiple choice quiz from study notes."""
        valid_notes = self._validate_text(notes, field_name="Study Notes for Quiz")
        if not (1 <= num_questions <= 10):
            raise ValueError("Number of questions must be between 1 and 10.")
        
        prompt = StudyPrompts.get_quiz_prompt(valid_notes, num_questions=num_questions, difficulty=difficulty)
        output = self.llm_service.generate_text(
            prompt=prompt,
            system_instruction="You are an examination specialist. Generate rigorous MCQs with accurate answer keys and explanations."
        )
        return {
            "quiz": output,
            "num_questions": num_questions,
            "difficulty": difficulty,
            "status": "success"
        }

    def explain_concept(self, concept: str, audience_level: str = "Beginner") -> Dict[str, Any]:
        """Generates clear, multi-faceted conceptual explanations."""
        valid_concept = self._validate_text(concept, field_name="Concept or Topic")
        prompt = StudyPrompts.get_concept_explainer_prompt(valid_concept, target_audience=audience_level)
        output = self.llm_service.generate_text(
            prompt=prompt,
            system_instruction="You are a renowned computer science and engineering professor who makes complex ideas simple through intuition and clear analogies."
        )
        return {
            "explanation": output,
            "concept": valid_concept,
            "audience_level": audience_level,
            "status": "success"
        }

    def improve_answer(self, draft_answer: str, question: str = "") -> Dict[str, Any]:
        """Improves and critiques a student's draft answer."""
        valid_draft = self._validate_text(draft_answer, field_name="Draft Answer")
        prompt = StudyPrompts.get_answer_improver_prompt(valid_draft, question=question)
        output = self.llm_service.generate_text(
            prompt=prompt,
            system_instruction="You are an academic writing evaluator. Enhance clarity, structure, and precision while providing constructive feedback."
        )
        return {
            "improved_answer": output,
            "original_chars": len(valid_draft),
            "status": "success"
        }
