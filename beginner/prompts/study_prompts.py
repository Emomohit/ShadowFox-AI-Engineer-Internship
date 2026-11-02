"""
Prompts module for StudyPilot AI.
All prompt templates are decoupled from UI and business logic.
"""
from typing import Dict


class StudyPrompts:
    """Prompt templates crafted specifically for student learning workflows."""

    @staticmethod
    def get_summarizer_prompt(text: str, summary_type: str = "detailed") -> str:
        instructions = {
            "concise": "Provide a high-impact, concise summary in 3-4 sentences, followed by key takeaways.",
            "detailed": "Provide a comprehensive, structured summary organized with major topics, mechanisms, and deep insights.",
            "revision": "Provide high-yield revision flashcard-style bullet points, highlighting key formulas, definitions, and rules."
        }
        selected_instruction = instructions.get(summary_type, instructions["detailed"])

        return f"""You are an elite academic tutor and subject matter expert.
Your task is to analyze the following student notes and produce an exceptionally structured, clear, and pedagogically sound summary.

SUMMARY FORMAT INSTRUCTION:
{selected_instruction}

ADDITIONAL REQUIREMENTS:
1. Identify and explicitly define ALL key technical terms, concepts, and definitions in a dedicated section titled "### 🔑 Core Concepts & Definitions".
2. Maintain academic rigor without unnecessary jargon.
3. Use bolding and structured markdown for effortless readability.

STUDENT NOTES:
\"\"\"
{text}
\"\"\"

SUMMARY:"""

    @staticmethod
    def get_quiz_prompt(text: str, num_questions: int = 5, difficulty: str = "Medium") -> str:
        return f"""You are an expert exam designer and educator.
Your task is to create a {difficulty}-level Multiple Choice Quiz ({num_questions} questions) strictly based on the provided study notes.

REQUIREMENTS:
1. Generate exactly {num_questions} questions.
2. For each question, provide 4 distinct options: A), B), C), D). Only ONE must be unequivocally correct.
3. Clearly label the correct answer (e.g. "**Correct Answer: B**").
4. Provide a detailed pedagogical explanation explaining WHY the correct option is right and why the plausible distractors are incorrect.
5. Do NOT reference outside topics not present or implied by the notes.

FORMAT EACH QUESTION AS FOLLOWS:
**Question [X]: [Question Text]**
- A) [Option A]
- B) [Option B]
- C) [Option C]
- D) [Option D]
**Correct Answer: [Letter]**
**Explanation:** [Detailed explanation referencing the notes]

STUDENT NOTES:
\"\"\"
{text}
\"\"\"

QUIZ:"""

    @staticmethod
    def get_concept_explainer_prompt(concept: str, target_audience: str = "Beginner") -> str:
        return f"""You are a master educator known for explaining complex technical topics clearly using intuition and vivid analogies.
Explain the concept: "{concept}" for a {target_audience} student.

STRUCTURE YOUR EXPLANATION AS:
1. ### 🎯 The Big Picture (ELI5 / Plain English intuition)
2. ### 🚗 Vivid Real-World Analogy (Compare it to everyday life)
3. ### ⚙️ How It Works (Technical breakdown with step-by-step logic)
4. ### ⚠️ Common Pitfalls & Misconceptions (What students often get wrong)
5. ### 📝 Quick Revision Checklist (3-4 memorable summary bullets)

Keep the tone encouraging, crystal clear, and intellectually engaging."""

    @staticmethod
    def get_answer_improver_prompt(student_answer: str, question: str = "") -> str:
        q_context = f"QUESTION / PROMPT:\n{question}\n\n" if question.strip() else ""
        return f"""You are a senior academic grading examiner and technical writing mentor.
A student has submitted a draft answer. Your job is to elevate this answer to university top-grade standard while preserving the student's authentic intent.

{q_context}STUDENT'S ORIGINAL DRAFT:
\"\"\"
{student_answer}
\"\"\"

PLEASE PROVIDE:
### 🌟 Improved Version (Exemplary, well-structured, clear, technically accurate)
### 📊 Comparative Analysis & What Changed:
- **Structure & Coherence:** How the paragraph and argument organization was enhanced.
- **Vocabulary & Precision:** Better terminology used in place of vague statements.
- **Accuracy & Depth:** Missing nuances or corrections made.
### 💡 Actionable Advice for Next Time (1-2 practical tips the student can apply)"""
