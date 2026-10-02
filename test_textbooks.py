"""
test_textbooks.py — Quick smoke-test of the Dual-PICO RAG pipeline.

Uses the Textbooks corpus (smallest, ~350 MB download, no GPU needed).
Run this to verify the pipeline works before scaling to PubMed/Wikipedia.

Usage:
    python test_textbooks.py [--mode dual|input_only|output_only|standard]
                             [--corpus Textbooks|PubMed|Wikipedia|StatPearls]
                             [--retriever MedCPT|Contriever|Specter]

Examples:
    python test_textbooks.py                        # default: dual + Textbooks
    python test_textbooks.py --mode standard        # vanilla RAG baseline
    python test_textbooks.py --corpus PubMed        # run on PubMed corpus
"""
import sys
import json
import argparse
from pathlib import Path

# Ensure the project root is on sys.path
sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline import DualPICORAG

# ── Sample test questions (mix of MCQ and Yes/No) ─────────────────────────────
TEST_QUESTIONS = [
    {
        "id":       "q1",
        "type":     "mcq",
        "question": (
            "A 45-year-old man with hypertension is started on a medication that "
            "blocks angiotensin-converting enzyme. Which of the following is the "
            "most common side effect of this drug class?"
        ),
        "options":  "A. Hyperkalemia\nB. Dry cough\nC. Angioedema\nD. Hyponatremia",
        "answer":   "B",
    },
    {
        "id":       "q2",
        "type":     "mcq",
        "question": (
            "Which antibiotic is the drug of choice for treating community-acquired "
            "pneumonia caused by atypical organisms such as Mycoplasma pneumoniae?"
        ),
        "options":  "A. Amoxicillin\nB. Vancomycin\nC. Azithromycin\nD. Ceftriaxone",
        "answer":   "C",
    },
    {
        "id":       "yn",
        "type":     "yn",
        "question": "Is metformin contraindicated in patients with severe renal impairment?",
        "options":  "",
        "answer":   "YES",
    },
]


def run_tests(
    mode:      str = "dual",
    corpus:    str = "Textbooks",
    retriever: str = "MedCPT",
):
    print("\n" + "=" * 60)
    print(f"  DUAL-PICO RAG TEST")
    print(f"  mode={mode} | corpus={corpus} | retriever={retriever}")
    print("=" * 60)

    pipeline = DualPICORAG(
        mode=mode,
        corpus_name=corpus,
        retriever_name=retriever,
        top_k_ret=15,
        top_k_rank=3,
        verbose=True,
    )

    results = []
    correct = 0

    for q in TEST_QUESTIONS:
        print(f"\n{'#' * 60}")
        print(f"  Question [{q['id']}] ({q['type'].upper()})")
        print(f"  {q['question'][:120]}...")
        if q["options"]:
            for line in q["options"].split("\n"):
                print(f"  {line}")
        print(f"{'#' * 60}")

        result = pipeline.run(query=q["question"], options=q["options"])

        pred  = result["answer"].strip().upper()
        label = q["answer"].strip().upper()
        # Correct if pred is non-empty and starts with (or equals) the expected label
        is_correct = bool(pred) and (pred.startswith(label) or label.startswith(pred))

        correct += is_correct
        status   = "✓ CORRECT" if is_correct else f"✗ WRONG  (expected {label})"
        print(f"  Predicted: {pred!r}   {status}")
        print(f"  Timings:   {result['timings']}")

        results.append({
            "id":        q["id"],
            "predicted": pred,
            "expected":  label,
            "correct":   is_correct,
            "timings":   result["timings"],
            "top_docs":  [
                {
                    "title": d["title"],
                    "score": d.get("combined_score", d["retrieval_score"]),
                }
                for d in result["top_docs"]
            ],
        })

    # ── Summary ───────────────────────────────────────────────────────────────
    total       = len(TEST_QUESTIONS)
    accuracy    = correct / total * 100
    avg_latency = sum(r["timings"]["t_total"] for r in results) / total

    print(f"\n{'=' * 60}")
    print(f"  RESULTS  [{mode} | {corpus} | {retriever}]")
    print(f"{'=' * 60}")
    print(f"  Accuracy:     {correct}/{total}  ({accuracy:.1f}%)")
    print(f"  Avg latency:  {avg_latency:.2f}s per question")
    print(f"{'=' * 60}\n")

    out_path = Path(__file__).parent / f"results_{mode}_{corpus}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "mode":         mode,
                "corpus":       corpus,
                "retriever":    retriever,
                "accuracy":     accuracy,
                "avg_latency_s": avg_latency,
                "details":      results,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"  Results saved → {out_path}")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Dual-PICO RAG pipeline")
    parser.add_argument(
        "--mode", default="dual",
        choices=["dual", "input_only", "output_only", "standard"],
        help="Pipeline mode / ablation (default: dual)",
    )
    parser.add_argument(
        "--corpus", default="Textbooks",
        choices=["Textbooks", "PubMed", "Wikipedia", "StatPearls"],
        help="MedRAG corpus to use (default: Textbooks)",
    )
    parser.add_argument(
        "--retriever", default="MedCPT",
        choices=["MedCPT", "Contriever", "Specter", "BM25"],
        help="Retriever encoder (default: MedCPT)",
    )
    args = parser.parse_args()
    run_tests(mode=args.mode, corpus=args.corpus, retriever=args.retriever)
