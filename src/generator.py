"""
generator.py — Stage 4: Evidence-Grounded Answer Generation (LLM Call #3)

Takes the top-3 PICO-aligned documents and the expanded query,
produces a final grounded answer (MCQ letter or Yes/No).
"""
from pathlib import Path
from src.llm import LLMClient

PROMPT_PATH     = Path(__file__).parent.parent / "prompts" / "generation_prompt.txt"
PROMPT_TEMPLATE = PROMPT_PATH.read_text(encoding="utf-8")

# Max characters per document snippet sent to the LLM.
# 900 chars ≈ ~225 tokens — enough context without blowing the context window.
DOC_CONTENT_LIMIT = 900


class Generator:
    """Generates a grounded answer from top-k evidence documents."""

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def generate(
        self,
        expanded_query: str,
        top_docs: list[dict],
        options: str = "",
    ) -> str:
        """
        Args:
            expanded_query: professional version of the question (from Stage 1)
            top_docs:       top-k docs from Stage 3 (each has title + content)
            options:        MCQ options string e.g. "A. Beta blocker  B. ACE inhibitor"
                            Leave empty for yes/no questions.

        Returns:
            Single token answer: "A" / "B" / "C" / "D"  or  "Yes" / "No"
        """
        # Build evidence block (up to 3 docs, capped per doc)
        evidence_block = ""
        for i, doc in enumerate(top_docs[:3], 1):
            snippet = doc.get("content", "")[:DOC_CONTENT_LIMIT]
            evidence_block += f"[Doc {i}] {doc.get('title', 'Untitled')}\n{snippet}\n\n"

        # Append MCQ options to query if provided
        full_query = expanded_query
        if options:
            full_query = f"{expanded_query}\n\nOptions:\n{options}"

        prompt = (
            PROMPT_TEMPLATE
            .replace("{expanded_query}", full_query)
            .replace("{evidence_block}", evidence_block.strip())
        )

        answer = self.llm.generate(prompt, max_tokens=32, temperature=0.0)
        # Return first whitespace-separated token (e.g. "A" from "A.")
        return answer.strip().split()[0] if answer.strip() else ""
