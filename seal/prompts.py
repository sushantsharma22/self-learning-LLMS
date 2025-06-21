"""
Prompt templates used for SEAL self-edits and (optionally) zero-shot evaluation.

The paper uses two families of tasks:
1. Knowledge incorporation (SQuAD-style QA)
2. Abstract reasoning (ARC multiple-choice)

We keep both prompt styles here so the outer loop can choose at runtime.
"""

# ---------- knowledge prompt ----------

KNOWLEDGE_PROMPT = """\
You are an expert question-answering system.
Improve the answer so that it is **concise, factually correct, and directly answers the question**.
--
QUESTION: {question}
CURRENT_ANSWER: {answer}
REFERENCE: {ground_truth}
--
Revised answer (no commentary, no extra text):
"""

def build_knowledge_prompt(question: str, answer: str, ground_truth: str) -> str:
    """Fill the SQuAD prompt."""
    return KNOWLEDGE_PROMPT.format(
        question=question,
        answer=answer,
        ground_truth=ground_truth,
    )

# ---------- ARC prompt ----------

ARC_PROMPT = """\
You are a scientific-reasoning assistant tackling an ARC multiple-choice puzzle.
Given the question, choose the correct option **only**.
--
QUESTION: {question}
OPTIONS:
{options}
CURRENT_CHOICE: {answer}
{explanation_section}--
Improved single-letter choice (A/B/C/D/E):
"""

def build_arc_prompt(
    question: str,
    options: str,
    answer: str,
    explanation: str | None = "",
) -> str:
    """
    Build the ARC prompt.

    The paper sometimes supplies the model's chain-of-thought (CoT) explanation.
    If `explanation` is empty, we omit that line to keep the prompt tight.
    """
    explanation_section = ""
    if explanation:
        explanation_section = f"EXPLANATION: {explanation}\n"
    return ARC_PROMPT.format(
        question=question,
        options=options,
        answer=answer,
        explanation_section=explanation_section,
    )
