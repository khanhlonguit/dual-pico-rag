"""
retriever.py — Dense retrieval backed by MedRAG's RetrievalSystem.

Supports all MedRAG corpora (Textbooks, PubMed, Wikipedia, StatPearls)
and all retrievers (MedCPT, Contriever, Specter, BM25).

Pre-computed embeddings are downloaded automatically from SharePoint
on first run (~350 MB for Textbooks, ~63 GB for PubMed, etc.).
Subsequent runs load the cached FAISS index in < 1 second.
"""
import os
import sys
from pathlib import Path

# ── Add MedRAG/src to Python path ─────────────────────────────────────────────
MEDRAG_SRC = Path(__file__).parent.parent / "MedRAG" / "src"
if not MEDRAG_SRC.exists():
    raise FileNotFoundError(
        f"MedRAG source not found at {MEDRAG_SRC}.\n"
        "Run: git clone https://github.com/gzxiong/MedRAG.git <project_root>/MedRAG"
    )
if str(MEDRAG_SRC) not in sys.path:
    sys.path.insert(0, str(MEDRAG_SRC))

from utils import RetrievalSystem  # noqa: E402  (MedRAG internal)

# Default corpus storage directory (can be overridden via CORPUS_DIR env var)
DEFAULT_CORPUS_DIR = Path(__file__).parent.parent / "corpus"


class MedRAGRetriever:
    """
    Thin wrapper around MedRAG's RetrievalSystem.

    On first run the system will:
      1. git-clone the corpus chunk files from HuggingFace datasets
      2. Download pre-computed embeddings from SharePoint (if available)
         or embed locally using the encoder model (fallback)
      3. Build a FAISS index and cache it to disk

    All subsequent runs load the cached index instantly.

    Args:
        corpus_name:    One of "Textbooks", "PubMed", "Wikipedia", "StatPearls"
        retriever_name: One of "MedCPT", "Contriever", "Specter", "BM25"
        top_k:          Default number of candidates to retrieve
        db_dir:         Root directory for corpus / index storage
                        (defaults to $CORPUS_DIR env var or ./corpus/)
    """

    def __init__(
        self,
        corpus_name: str = "Textbooks",
        retriever_name: str = "MedCPT",
        top_k: int = 15,
        db_dir: str | None = None,
    ):
        self.corpus_name    = corpus_name
        self.retriever_name = retriever_name
        self.top_k          = top_k

        db_dir = db_dir or os.getenv("CORPUS_DIR", str(DEFAULT_CORPUS_DIR))
        os.makedirs(db_dir, exist_ok=True)

        print(f"[Retriever] corpus={corpus_name} | retriever={retriever_name}")
        print(f"[Retriever] db_dir={db_dir}")

        self._system = RetrievalSystem(
            retriever_name=retriever_name,
            corpus_name=corpus_name,
            db_dir=db_dir,
            HNSW=False,
            cache=False,
        )
        print("[Retriever] Ready.\n")

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict]:
        """
        Retrieve top-k documents for a query.

        Returns:
            List of dicts with keys: id, title, content, retrieval_score
        """
        k = top_k if top_k is not None else self.top_k
        texts, scores = self._system.retrieve(query, k=k)
        return [
            {
                "id":              doc.get("id", ""),
                "title":           doc.get("title", ""),
                "content":         doc.get("content", ""),
                "retrieval_score": float(score),
            }
            for doc, score in zip(texts, scores)
        ]
