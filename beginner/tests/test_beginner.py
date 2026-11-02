"""
Automated Unit and Integration Tests for Beginner Level: StudyPilot AI.
Verifies input validation, prompt construction, error handling, and end-to-end workflows.
"""
import pytest
import os
import sys

# Ensure root path is accessible
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from beginner.prompts.study_prompts import StudyPrompts
from beginner.services.llm_service import LLMService, LLMServiceError
from beginner.services.study_service import StudyService


@pytest.fixture
def mock_study_service():
    """Provides a StudyService instance wired to the deterministic mock provider."""
    llm = LLMService(use_mock=True)
    return StudyService(llm_service=llm)


# ==============================================================================
# 1. PROMPT GENERATION TESTS
# ==============================================================================
def test_summarizer_prompt_structure():
    prompt = StudyPrompts.get_summarizer_prompt("Operating systems manage hardware.", summary_type="detailed")
    assert "Operating systems manage hardware." in prompt
    assert "Core Concepts & Definitions" in prompt
    assert "detailed" in prompt.lower() or "comprehensive" in prompt.lower()


def test_quiz_prompt_structure():
    prompt = StudyPrompts.get_quiz_prompt("Binary Search operates on sorted arrays.", num_questions=3, difficulty="Hard")
    assert "Binary Search operates on sorted arrays." in prompt
    assert "3 questions" in prompt or "exactly 3" in prompt
    assert "Hard-level" in prompt or "Hard" in prompt
    assert "Correct Answer:" in prompt


def test_concept_explainer_prompt_structure():
    prompt = StudyPrompts.get_concept_explainer_prompt("Virtual Memory", target_audience="Beginner")
    assert "Virtual Memory" in prompt
    assert "Beginner" in prompt
    assert "Real-World Analogy" in prompt


def test_answer_improver_prompt_structure():
    prompt = StudyPrompts.get_answer_improver_prompt(
        student_answer="Process is running program. Thread is lightweight process.",
        question="Difference between Process and Thread"
    )
    assert "Process is running program" in prompt
    assert "Difference between Process and Thread" in prompt
    assert "Comparative Analysis" in prompt


# ==============================================================================
# 2. INPUT VALIDATION TESTS
# ==============================================================================
def test_empty_notes_validation(mock_study_service):
    with pytest.raises(ValueError, match="cannot be empty"):
        mock_study_service.summarize_notes("")

    with pytest.raises(ValueError, match="cannot be empty"):
        mock_study_service.summarize_notes("   ")


def test_short_input_validation(mock_study_service):
    with pytest.raises(ValueError, match="too short"):
        mock_study_service.summarize_notes("Hi")


def test_excessive_input_validation(mock_study_service):
    huge_text = "word " * 4000  # > 15,000 chars
    with pytest.raises(ValueError, match="exceeds the maximum limit"):
        mock_study_service.summarize_notes(huge_text)


def test_invalid_quiz_question_count(mock_study_service):
    with pytest.raises(ValueError, match="between 1 and 10"):
        mock_study_service.generate_quiz("Valid study notes content here.", num_questions=0)

    with pytest.raises(ValueError, match="between 1 and 10"):
        mock_study_service.generate_quiz("Valid study notes content here.", num_questions=15)


# ==============================================================================
# 3. SERVICE WORKFLOW & DETERMINISTIC GENERATION TESTS
# ==============================================================================
def test_summarize_workflow_success(mock_study_service):
    sample_notes = "Deadlock happens when processes hold resources and wait for others."
    res = mock_study_service.summarize_notes(sample_notes, summary_type="concise")
    assert res["status"] == "success"
    assert "Summary" in res["summary"]
    assert res["char_count"] == len(sample_notes)


def test_quiz_workflow_success(mock_study_service):
    sample_notes = "QuickSort is a divide and conquer algorithm choosing a pivot element."
    res = mock_study_service.generate_quiz(sample_notes, num_questions=2, difficulty="Medium")
    assert res["status"] == "success"
    assert res["num_questions"] == 2
    assert "Question" in res["quiz"]
    assert "Correct Answer:" in res["quiz"]


def test_concept_explain_workflow_success(mock_study_service):
    res = mock_study_service.explain_concept("TCP Handshake", audience_level="Beginner")
    assert res["status"] == "success"
    assert "Analogy" in res["explanation"] or "Plain English" in res["explanation"]


def test_answer_improver_workflow_success(mock_study_service):
    res = mock_study_service.improve_answer(
        draft_answer="HTTP is stateless because it does not remember past requests.",
        question="Why is HTTP called stateless?"
    )
    assert res["status"] == "success"
    assert "Improved" in res["improved_answer"]


# ==============================================================================
# 4. ERROR HANDLING & CREDENTIALS VERIFICATION
# ==============================================================================
def test_missing_api_key_error_in_live_mode():
    """Verify that when live mode is requested without a valid key, LLMServiceError is raised."""
    with pytest.raises(LLMServiceError, match="Missing valid Gemini API key"):
        LLMService(api_key="", provider="gemini", use_mock=False)


def test_empty_prompt_error():
    llm = LLMService(use_mock=True)
    with pytest.raises(ValueError, match="Prompt cannot be empty"):
        llm.generate_text("")
