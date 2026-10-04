import sys
import argparse
import json
from tqdm import tqdm
from datasets import load_dataset
from src.pipeline import DualPICORAG

def evaluate_mirage(corpus="Textbooks", retriever="MedCPT", dataset_name="pubmedqa", limit=10):
    """
    Run MIRAGE benchmark on the given corpus.
    """
    print(f"Loading {dataset_name} dataset...")
    
    eval_data = []
    if dataset_name == "pubmedqa":
        # PubMedQA has yes/no/maybe answers
        ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
        for item in ds:
            question = item["question"]
            answer = item["final_decision"] # "yes", "no", or "maybe"
            options_str = "A. yes\nB. no\nC. maybe"
            answer_idx = "a" if answer == "yes" else "b" if answer == "no" else "c"
            eval_data.append({"question": question, "options_str": options_str, "answer_idx": answer_idx})
            
    elif dataset_name == "medqa":
        # MedQA-USMLE 4-options
        ds = load_dataset("GBaker/MedQA-USMLE-4-options", split="test")
        for item in ds:
            question = item["question"]
            options = item["options"]
            options_str = "\n".join([f"{k}. {v}" for k, v in options.items()])
            answer_idx = item["answer_idx"].lower()
            eval_data.append({"question": question, "options_str": options_str, "answer_idx": answer_idx})
            
    else:
        raise ValueError("Unsupported dataset. Try pubmedqa or medqa")

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
        options_str = item["options_str"]
        true_ans_idx = item["answer_idx"]
        
        # Dual-PICO pipeline
        response_dict = rag.run(q, options=options_str)
        raw_pred = response_dict["answer"].lower().strip()
        
        # Llama 3 sẽ trả về A, B, C, hoặc D (có thể kèm dấu chấm, vd "a.")
        pred_ans_idx = ""
        if raw_pred.startswith("a"):
            pred_ans_idx = "a"
        elif raw_pred.startswith("b"):
            pred_ans_idx = "b"
        elif raw_pred.startswith("c"):
            pred_ans_idx = "c"
        elif raw_pred.startswith("d"):
            pred_ans_idx = "d"
        else:
            pred_ans_idx = raw_pred # fallback nếu LLM nói nhảm
        
        # Evaluate
        is_correct = (true_ans_idx == pred_ans_idx)
        if is_correct:
            correct += 1
            
        results.append({
            "question": q,
            "true_answer": true_ans_idx,
            "pred_answer": pred_ans_idx,
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
    parser.add_argument("--dataset", type=str, default="medqa", choices=["pubmedqa", "medqa"])
    parser.add_argument("--limit", type=int, default=10, help="Number of questions to test (default 10 for quick test)")
    args = parser.parse_args()
    
    evaluate_mirage(corpus=args.corpus, dataset_name=args.dataset, limit=args.limit)
