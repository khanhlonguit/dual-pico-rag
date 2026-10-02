"""
pico_rewriter.py — Stage 1: PICO-Supported Query Rewriting (LLM Call #1)

Three sub-steps:
  1. Discipline classification
  2. Professional terminology expansion
  3. P/I/C/O decomposition

Output: structured PICO dict + flattened query string for embedding.
"""
import json
import re
from pathlib import Path
from src.llm import LLMClient

PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "pico_rewrite_prompt.txt"
PROMPT_TEMPLATE = PROMPT_PATH.read_text(encoding="utf-8")


def _extract_json(text: str) -> dict:
    """Robustly extract the first JSON object from an LLM response."""
    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try extracting JSON block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {}


class PICORewriter:
    """Rewrites a raw clinical query into a structured PICO representation."""

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def rewrite(self, query: str) -> dict:
        """
        Args:
            query: raw clinical question from user

        Returns dict with:
            - discipline      (str)
            - expanded_query  (str)  — professional terminology version
            - pico            (dict) — {P, I, C, O}
            - structured_query(str) — flattened string for embedding
        """
        prompt = PROMPT_TEMPLATE.replace("{query}", query)
        raw    = self.llm.generate(prompt, max_tokens=512, temperature=0.0)
        result = _extract_json(raw)

        # Fallback if parsing failed
        if not result or "pico" not in result:
            print(f"[Rewriter] Warning: JSON parse failed. Using raw query.")
            return {
                "discipline":       "General Medicine",
                "expanded_query":   query,
                "pico":             {"P": "Not specified", "I": "Not specified",
                                     "C": "Not specified", "O": "Not specified"},
                "structured_query": query,
            }

        pico = result.get("pico", {})

        # Build flat structured query for embedding
        parts = [result.get("expanded_query", query)]
        for label, key in [("Population", "P"), ("Intervention", "I"),
                            ("Comparison", "C"),  ("Outcome", "O")]:
            val = pico.get(key, "Not specified")
            if val and val.lower() != "not specified":
                parts.append(f"{label}: {val}")

        result["structured_query"] = ". ".join(parts)
        return result
