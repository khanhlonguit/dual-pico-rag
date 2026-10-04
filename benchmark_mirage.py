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
    # Khởi tạo DualPICORAG với mode "dual"
    rag = DualPICORAG(mode="dual", retriever_name=retriever, corpus_name=corpus)

    correct = 0
    results = []
    print(f"\nRunning Benchmark on {len(eval_data)} questions...")
    for item in tqdm(eval_data):
        q = item["question"]
        true_ans = item["answer"].lower()
        
        # Cung cấp options rõ ràng để ép LLM trả về A, B, hoặc C
        options_str = "A. yes\nB. no\nC. maybe"
        
        # Dual-PICO pipeline
        response_dict = rag.run(q, options=options_str)
        raw_pred = response_dict["answer"].lower().strip()
        
        # Llama 3 sẽ trả về A, B, hoặc C (có thể kèm dấu chấm, vd "a.")
        pred_ans = ""
        if raw_pred.startswith("a"):
            pred_ans = "yes"
        elif raw_pred.startswith("b"):
            pred_ans = "no"
        elif raw_pred.startswith("c"):
            pred_ans = "maybe"
        else:
            pred_ans = raw_pred # fallback nếu LLM nói nhảm
        
        # Evaluate
        is_correct = (true_ans == pred_ans)
        if is_correct:
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
