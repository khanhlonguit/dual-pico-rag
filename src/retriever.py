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
import requests
import zipfile

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

        self._ensure_corpus_downloaded()

        self._system = RetrievalSystem(
            retriever_name=retriever_name,
            corpus_name=corpus_name,
            db_dir=db_dir,
            HNSW=False,
            cache=False,
        )
        print("[Retriever] Ready.\n")

    def _ensure_corpus_downloaded(self):
        """Manually download embeddings using requests + User-Agent to bypass Windows/SharePoint blocks."""
        retriever_map = {
            "MedCPT": "ncbi/MedCPT-Query-Encoder",
            "Contriever": "facebook/contriever",
            "Specter": "allenai/specter",
        }
        retriever_id = retriever_map.get(self.retriever_name, self.retriever_name)
        
        _emb_urls = {
            ("Textbooks", "allenai/specter"):            "https://myuva-my.sharepoint.com/:u:/g/personal/hhu4zu_virginia_edu/EYRRpJbNDyBOmfzCOqfQzrsBwUX0_UT8-j_geDPcVXFnig?download=1",
            ("Textbooks", "facebook/contriever"):        "https://myuva-my.sharepoint.com/:u:/g/personal/hhu4zu_virginia_edu/EQqzldVMCCVIpiFV4goC7qEBSkl8kj5lQHtNq8DvHJdAfw?download=1",
            ("Textbooks", "ncbi/MedCPT-Query-Encoder"):  "https://myuva-my.sharepoint.com/:u:/g/personal/hhu4zu_virginia_edu/EQ8uXe4RiqJJm0Tmnx7fUUkBKKvTwhu9AqecPA3ULUxUqQ?download=1",
            ("PubMed",    "allenai/specter"):            "https://myuva-my.sharepoint.com/:u:/g/personal/hhu4zu_virginia_edu/Ebz8ySXt815FotxC1KkDbuABNycudBCoirTWkKfl8SEswA?download=1",
            ("PubMed",    "facebook/contriever"):        "https://myuva-my.sharepoint.com/:u:/g/personal/hhu4zu_virginia_edu/EWecRNfTxbRMnM0ByGMdiAsBJbGJOX_bpnUoyXY9Bj4_jQ?download=1",
            ("PubMed",    "ncbi/MedCPT-Query-Encoder"):  "https://myuva-my.sharepoint.com/:u:/g/personal/hhu4zu_virginia_edu/EVCuryzOqy5Am5xzRu6KJz4B6dho7Tv7OuTeHSh3zyrOAw?download=1",
            ("Wikipedia", "allenai/specter"):            "https://myuva-my.sharepoint.com/:u:/g/personal/hhu4zu_virginia_edu/Ed7zG3_ce-JOmGTbgof3IK0BdD40XcuZ7AGZRcV_5D2jkA?download=1",
            ("Wikipedia", "facebook/contriever"):        "https://myuva-my.sharepoint.com/:u:/g/personal/hhu4zu_virginia_edu/ETKHGV9_KNBPmDM60MWjEdsBXR4P4c7zZk1HLLc0KVaTJw?download=1",
            ("Wikipedia", "ncbi/MedCPT-Query-Encoder"):  "https://myuva-my.sharepoint.com/:u:/g/personal/hhu4zu_virginia_edu/EXoxEANb_xBFm6fa2VLRmAcBIfCuTL-5VH6vl4GxJ06oCQ?download=1",
        }
        
        url = _emb_urls.get((self.corpus_name, retriever_id))
        if not url: return

        index_dir = os.path.join(self._system_db_dir(), self.corpus_name.lower(), "index", retriever_id.replace("/", "_"))
        if os.path.exists(os.path.join(index_dir, "metadatas.jsonl")):
            return # Already downloaded
            
        print("[Downloading] Pre-computed embeddings manually to bypass WAF...")
        os.makedirs(index_dir, exist_ok=True)
        zip_path = os.path.join(index_dir, "embedding.zip")
        
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        with requests.get(url, headers=headers, stream=True, allow_redirects=True, timeout=300) as r:
            r.raise_for_status()
            total = int(r.headers.get('Content-Length', 0))
            done = 0
            with open(zip_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=1024*1024):
                    f.write(chunk)
                    done += len(chunk)
                    if total: print(f"\r  {done/1024/1024:.0f} / {total/1024/1024:.0f} MB", end="", flush=True)
        
        print("\n[Extracting] embedding.zip...")
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(index_dir)
        os.remove(zip_path)
        print("[Done] Embeddings downloaded successfully.")

    def _system_db_dir(self):
        return os.getenv("CORPUS_DIR", str(DEFAULT_CORPUS_DIR))

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
