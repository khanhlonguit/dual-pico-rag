"""
pico_reranker.py — Stage 3: PICO-Supported Re-ranking via MedCPT Cross-Encoder

This replaces the slow, unstable LLM-based CoT reranker with the state-of-the-art
ncbi/MedCPT-Cross-Encoder. It receives the PICO-structured query and scores 
the top retrieved candidates directly using the cross-encoder logic.
"""
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.llm import LLMClient

class PICOReranker:
    """Re-ranks retrieved documents using the MedCPT Cross-Encoder, guided by the PICO-structured query."""

    def __init__(self, llm: LLMClient):
        # The 'llm' argument is kept for compatibility with pipeline.py but unused.
        self.model_name = "ncbi/MedCPT-Cross-Encoder"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Reranker] Loading {self.model_name} on {self.device}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name).to(self.device)
        self.model.eval()

    def rerank(self, structured_query: str, candidates: list[dict], top_k: int = 3) -> list[dict]:
        """
        Args:
            structured_query: The flattened PICO query string from Stage 1
            candidates:       top docs from retriever (each has 'title' and 'content')
            top_k:            number of docs to keep

        Returns: top_k docs sorted by cross-encoder score.
        """
        if not candidates:
            return []

        # Format pairs for cross-encoder: [query, title + "[SEP]" + content]
        pairs = []
        for doc in candidates:
            title = doc.get("title", "")
            content = doc.get("content", "")
            # MedCPT uses title + [SEP] + abstract format
            doc_text = f"{title}[SEP]{content}"
            pairs.append([structured_query, doc_text])

        # Inference
        with torch.no_grad():
            encoded = self.tokenizer(
                pairs,
                truncation=True,
                padding=True,
                return_tensors="pt",
                max_length=512,
            ).to(self.device)
            
            logits = self.model(**encoded).logits.squeeze(-1)
            
            # Convert to list. If only one candidate, logits might be a scalar
            if logits.dim() == 0:
                scores = [logits.item()]
            else:
                scores = logits.cpu().numpy().tolist()

        scored = []
        for i, (doc, score) in enumerate(zip(candidates, scores)):
            doc = doc.copy()
            doc["cross_encoder_score"] = float(score)
            doc["combined_score"] = float(score)  # Pure Cross-Encoder sorting
            scored.append(doc)
            
            print(f"  [Reranker] Doc {i+1:>2}/{len(candidates)} "
                  f"| retrieval={doc.get('retrieval_score', 0.0):.3f} "
                  f"| cross-encoder={score:.3f}")

        # Sort by cross-encoder score descending
        scored.sort(key=lambda d: d["combined_score"], reverse=True)
        return scored[:top_k]
