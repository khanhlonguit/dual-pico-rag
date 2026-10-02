"""
pico_reranker.py — Stage 3: PICO-Supported Re-ranking via Chain-of-Thought (LLM Call #2)

For each of the top-15 retrieved candidates:
  1. Extract PICO elements from the document (CoT)
  2. Compare element-by-element with the query's PICO
  3. Assign a PICO-alignment score (0.0 – 1.0)
  4. Combine with retrieval cosine similarity
  5. Return top-3 most clinically aligned documents
"""
import json
import re
from pathlib import Path
from src.llm import LLMClient

PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "pico_rerank_prompt.txt"
PROMPT_TEMPLATE = PROMPT_PATH.read_text(encoding="utf-8")

# Combination weights (α · retrieval_score + β · pico_alignment_score)
ALPHA = 0.5
BETA  = 0.5


def _extract_json(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return {}


class PICOReranker:
    """Re-ranks retrieved documents by PICO alignment using CoT prompting."""

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def _score_one(self, query_pico: dict, doc: dict) -> dict:
        """Score a single document against the query PICO."""
        prompt = (
            PROMPT_TEMPLATE
            .replace("{P}", query_pico.get("P", "Not specified"))
            .replace("{I}", query_pico.get("I", "Not specified"))
            .replace("{C}", query_pico.get("C", "Not specified"))
            .replace("{O}", query_pico.get("O", "Not specified"))
            .replace("{doc_title}",   doc.get("title",   "")[:200])
            .replace("{doc_content}", doc.get("content", "")[:1200])
        )
        raw    = self.llm.generate(prompt, max_tokens=400, temperature=0.0)
        result = _extract_json(raw)
        return result

    def rerank(self, query_pico: dict, candidates: list[dict],
               top_k: int = 3) -> list[dict]:
        """
        Args:
            query_pico:  {P, I, C, O} from Stage 1 rewriting
            candidates:  top-15 docs from retriever (each has 'retrieval_score')
            top_k:       number of docs to keep (paper uses 3)

        Returns: top_k docs sorted by combined score, each with extra fields:
            - pico_alignment_score
            - combined_score
            - pico_reasoning
        """
        scored = []
        for i, doc in enumerate(candidates):
            result   = self._score_one(query_pico, doc)
            pa_score = float(result.get("pico_alignment_score", 0.0))
            # Clamp to [0, 1]
            pa_score = max(0.0, min(1.0, pa_score))

            combined = (ALPHA * doc.get("retrieval_score", 0.0)
                        + BETA  * pa_score)

            doc = doc.copy()
            doc["pico_alignment_score"] = pa_score
            doc["combined_score"]       = combined
            doc["pico_reasoning"]       = result.get("reasoning", "")
            doc["doc_pico"]             = result.get("doc_pico", {})
            scored.append(doc)

            print(f"  [Reranker] Doc {i+1:>2}/{len(candidates)} "
                  f"| retrieval={doc['retrieval_score']:.3f} "
                  f"| PICO={pa_score:.3f} "
                  f"| combined={combined:.3f}")

        scored.sort(key=lambda d: d["combined_score"], reverse=True)
        return scored[:top_k]
