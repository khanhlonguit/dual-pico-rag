import sys
import argparse
import json
from tqdm import tqdm
from datasets import load_dataset
from src.pipeline import DualPICORAG

def evaluate_mirage(corpus="Textbooks", retriever="MedCPT", dataset_name="pubmedqa", limit=10):
    """
    Run MIRAGE benchmark on the given corpus.
    For demonstration, we use PubMedQA (pqa_labeled) which is part of MIRAGE.
    """
    print(f"Loading {dataset_name} dataset...")
    if dataset_name == "pubmedqa":
        # PubMedQA has yes/no/maybe answers
        ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
        # Extract question and answer
        eval_data = []
        for item in ds:
            question = item["question"]
            answer = item["final_decision"] # "yes", "no", or "maybe"
            eval_data.append({"question": question, "answer": answer})
    else:
        raise ValueError("Unsupported dataset. Try pubmedqa")

    if limit:
        eval_data = eval_data[:limit]

    print(f"Initializing Dual-PICO RAG (Corpus: {corpus}, Retriever: {retriever})...")
    # Tự động chọn standard mode nếu muốn so sánh, ở đây mặc định chạy dual
    rag = DualPICORAG(retriever_name=retriever, corpus_name=corpus)

    correct = 0
    results = []
    print(f"\nRunning Benchmark on {len(eval_data)} questions...")
    for item in tqdm(eval_data):
        q = item["question"]
        true_ans = item["answer"].lower()
        
        # Dual-PICO pipeline
        response = rag.run(q, ablation_mode="dual")
        pred_ans = response.lower()
        
        # Đánh giá đơn giản cho pubmedqa (yes/no/maybe)
        is_correct = False
        if true_ans in pred_ans:
            is_correct = True
            correct += 1
            
        results.append({
            "question": q,
            "true_answer": true_ans,
            "pred_answer": pred_ans,
            "is_correct": is_correct
        })
        
    acc = correct / len(eval_data)
    print(f"\n--- Benchmark Results ---")
    print(f"Accuracy: {acc*100:.2f}% ({correct}/{len(eval_data)})")
    
    with open(f"mirage_{dataset_name}_{corpus}_results.json", "w") as f:
        json.dump(results, f, indent=4)
    print(f"Saved detailed results to mirage_{dataset_name}_{corpus}_results.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=str, default="Textbooks", choices=["Textbooks", "PubMed", "Wikipedia", "StatPearls"])
    parser.add_argument("--dataset", type=str, default="pubmedqa", choices=["pubmedqa"])
    parser.add_argument("--limit", type=int, default=10, help="Number of questions to test (default 10 for quick test)")
    args = parser.parse_args()
    
    evaluate_mirage(corpus=args.corpus, dataset_name=args.dataset, limit=args.limit)
