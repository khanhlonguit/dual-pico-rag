"""
pipeline.py — Full Dual-PICO RAG pipeline

Implements the MedCoT-RAG (Dual-PICO RAG) paper with 4 ablation modes:

  "dual"        → Full proposal: PICO rewriting (Stage 1) + PICO reranking (Stage 3)
  "input_only"  → Ablation: only PICO rewriting, no reranking
  "output_only" → Ablation: only PICO reranking, no rewriting
  "standard"    → Baseline: vanilla RAG, no PICO anywhere
"""
import time
from src.llm           import LLMClient
from src.pico_rewriter import PICORewriter
from src.retriever     import MedRAGRetriever
from src.pico_reranker import PICOReranker
from src.generator     import Generator

VALID_MODES = ("dual", "input_only", "output_only", "standard")


class DualPICORAG:
    """
    End-to-end Dual-PICO RAG pipeline.

    Args:
        mode:           ablation mode (see module docstring)
        corpus_name:    MedRAG corpus — "Textbooks", "PubMed", "Wikipedia", "StatPearls"
        retriever_name: encoder — "MedCPT", "Contriever", "Specter", "BM25"
        top_k_ret:      candidates retrieved from FAISS (default 15)
        top_k_rank:     docs kept after PICO reranking (default 3)
        llm_model:      Groq model name (overrides GROQ_MODEL env var)
        verbose:        print stage-by-stage progress
    """

    def __init__(
        self,
        mode:           str  = "dual",
        corpus_name:    str  = "Textbooks",
        retriever_name: str  = "MedCPT",
        top_k_ret:      int  = 15,
        top_k_rank:     int  = 3,
        llm_model:      str | None = None,
        verbose:        bool = True,
    ):
        if mode not in VALID_MODES:
            raise ValueError(f"mode must be one of {VALID_MODES}, got '{mode}'")

        self.mode       = mode
        self.top_k_ret  = top_k_ret
        self.top_k_rank = top_k_rank
        self.verbose    = verbose

        print(f"\n{'='*55}")
        print(f"  Dual-PICO RAG  [mode={mode}]")
        print(f"  corpus={corpus_name} | retriever={retriever_name}")
        print(f"{'='*55}")

        llm_kwargs = {"verbose": verbose}
        if llm_model:
            llm_kwargs["model"] = llm_model

        self.llm       = LLMClient(**llm_kwargs)
        self.rewriter  = PICORewriter(self.llm)
        self.retriever = MedRAGRetriever(
            corpus_name=corpus_name,
            retriever_name=retriever_name,
            top_k=top_k_ret,
        )
        self.reranker  = PICOReranker(self.llm)
        self.generator = Generator(self.llm)

    # ── Main entry point ──────────────────────────────────────────────────────

    def run(self, query: str, options: str = "") -> dict:
        """
        Run the full pipeline on one clinical question.

        Args:
            query:   raw clinical question
            options: MCQ options string (empty for yes/no questions)

        Returns dict:
            query, expanded_query, pico, top_docs, answer, timings
        """
        timings = {}
        t0      = time.perf_counter()

        # ── Stage 1: PICO Query Rewriting ────────────────────────────────────
        if self.verbose:
            print("\n[Stage 1] PICO Query Rewriting...")

        if self.mode in ("dual", "input_only"):
            t1             = time.perf_counter()
            rewrite        = self.rewriter.rewrite(query)
            timings["t_rewrite"] = round(time.perf_counter() - t1, 3)

            search_query   = rewrite["structured_query"]
            expanded_query = rewrite["expanded_query"]
            query_pico     = rewrite["pico"]

            if self.verbose:
                _sp = lambda x: str(x).encode("utf-8", errors="replace").decode("utf-8")
                print(f"  Discipline:  {_sp(rewrite['discipline'])}")
                print(f"  Expanded:    {_sp(expanded_query[:120])}")
                print(f"  PICO-P:      {_sp(query_pico['P'])}")
                print(f"  PICO-I:      {_sp(query_pico['I'])}")
                print(f"  PICO-C:      {_sp(query_pico['C'])}")
                print(f"  PICO-O:      {_sp(query_pico['O'])}")
        else:
            search_query   = query
            expanded_query = query
            query_pico     = None
            timings["t_rewrite"] = 0.0

        # ── Stage 2: Dense Retrieval ──────────────────────────────────────────
        if self.verbose:
            print(f"\n[Stage 2] Retrieving top-{self.top_k_ret} candidates...")

        t2          = time.perf_counter()
        candidates  = self.retriever.retrieve(search_query, top_k=self.top_k_ret)
        timings["t_retrieval"] = round(time.perf_counter() - t2, 3)

        if self.verbose and candidates:
            print(f"  Retrieved {len(candidates)} candidates "
                  f"(top score: {candidates[0]['retrieval_score']:.3f})")

        # ── Stage 3: PICO-Supported Re-ranking ───────────────────────────────
        if self.mode in ("dual", "output_only"):
            if self.verbose:
                print(f"\n[Stage 3] PICO Re-ranking {len(candidates)} candidates...")

            # output_only: rewriting was skipped, so extract PICO now
            if query_pico is None:
                rewrite        = self.rewriter.rewrite(query)
                query_pico     = rewrite["pico"]
                expanded_query = rewrite["expanded_query"]

            t3       = time.perf_counter()
            top_docs = self.reranker.rerank(
                query_pico, candidates, top_k=self.top_k_rank
            )
            timings["t_reranking"] = round(time.perf_counter() - t3, 3)
        else:
            top_docs = candidates[: self.top_k_rank]
            timings["t_reranking"] = 0.0

        if self.verbose:
            print(f"\n  Top-{self.top_k_rank} docs after reranking:")
            for i, d in enumerate(top_docs, 1):
                score = d.get("combined_score", d["retrieval_score"])
                print(f"    [{i}] {d['title'][:70]} (score={score:.3f})")

        # ── Stage 4: Evidence-Grounded Generation ────────────────────────────
        if self.verbose:
            print("\n[Stage 4] Generating answer...")

        t4      = time.perf_counter()
        answer  = self.generator.generate(expanded_query, top_docs, options)
        timings["t_generation"] = round(time.perf_counter() - t4, 3)
        timings["t_total"]      = round(time.perf_counter() - t0, 3)

        if self.verbose:
            print(f"\n{'─'*55}")
            print(f"  ANSWER: {answer}")
            print(f"  Total latency: {timings['t_total']}s")
            print(f"{'─'*55}\n")

        return {
            "query":          query,
            "expanded_query": expanded_query,
            "pico":           query_pico,
            "top_docs":       top_docs,
            "answer":         answer,
            "timings":        timings,
        }
