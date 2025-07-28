def build_knowledge_prompt(question: str) -> str:
    return (
        "<|user|>\n"
        f"Question: {question}\n"
        "Answer concisely and directly.\n"
        "<|assistant|>\n"
    )

# ARC prompt as before
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
    explanation_section = f"EXPLANATION: {explanation}\n" if explanation else ""
    return ARC_PROMPT.format(
        question=question,
        options=options,
        answer=answer,
        explanation_section=explanation_section,
    )
