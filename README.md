# Dual-PICO RAG (MedCoT-RAG)

Implementation of the **Dual-PICO RAG** pipeline from the MedCoT-RAG paper, built on top of the [MedRAG](https://github.com/gzxiong/MedRAG) toolkit.

## Architecture

```
Clinical Question
      │
      ▼
[Stage 1] PICO Query Rewriting          ← LLM call #1
  Classify discipline, expand terminology,
  decompose into P/I/C/O elements
      │
      ▼
[Stage 2] Dense Retrieval               ← MedCPT / FAISS
  Retrieve top-15 candidate documents
  from the medical corpus
      │
      ▼
[Stage 3] PICO Re-ranking               ← LLM call #2 × 15
  Chain-of-Thought scoring of each doc
  against the query PICO elements
      │
      ▼
[Stage 4] Answer Generation             ← LLM call #3
  Generate grounded MCQ/Yes-No answer
  from top-3 PICO-aligned documents
```

## Quick Start

### 1. Clone this repo and MedRAG

```bash
git clone <this_repo> dual_pico_rag
cd dual_pico_rag
git clone https://github.com/gzxiong/MedRAG.git MedRAG
```

### 2. Install dependencies

```bash
# CPU-only (laptop)
pip install -r requirements.txt

# GPU server (RTX 5080 / A100 etc.) — replace faiss-cpu with faiss-gpu
pip install -r requirements.txt
pip uninstall faiss-cpu -y
pip install faiss-gpu
```

### 3. Configure API key

```bash
cp .env.example .env
# Edit .env and set your GROQ_API_KEY
# Get a free key at https://console.groq.com
```

### 4. Run the smoke test

```bash
# Default: Dual-PICO mode on Textbooks corpus
python test_textbooks.py

# Other modes (ablation study)
python test_textbooks.py --mode standard     # vanilla RAG baseline
python test_textbooks.py --mode input_only   # PICO rewriting only
python test_textbooks.py --mode output_only  # PICO reranking only

# Other corpora (auto-downloads embeddings from SharePoint)
python test_textbooks.py --corpus PubMed
python test_textbooks.py --corpus Wikipedia
```

## Corpus Setup (Server)

Pre-computed embeddings are downloaded automatically on first run.

| Corpus     | Size      | Time (100 Mbps) |
|------------|-----------|-----------------|
| Textbooks  | ~350 MB   | ~30 sec         |
| PubMed     | ~63 GB    | ~1.5 hrs        |
| Wikipedia  | ~79 GB    | ~2 hrs          |
| StatPearls | ~few GB   | ~few min        |

Set `CORPUS_DIR` in `.env` to point to a large SSD on the server:

```env
CORPUS_DIR=/mnt/ssd/medrag_corpus
```

## Project Structure

```
dual_pico_rag/
├── MedRAG/                  # MedRAG toolkit (git clone separately)
├── corpus/                  # Downloaded embeddings (gitignored)
├── prompts/
│   ├── pico_rewrite_prompt.txt
│   ├── pico_rerank_prompt.txt
│   └── generation_prompt.txt
├── src/
│   ├── llm.py               # Groq API client
│   ├── retriever.py         # MedRAGRetriever (FAISS wrapper)
│   ├── pico_rewriter.py     # Stage 1: PICO Query Rewriting
│   ├── pico_reranker.py     # Stage 3: PICO Re-ranking (CoT)
│   ├── generator.py         # Stage 4: Answer Generation
│   └── pipeline.py          # DualPICORAG (end-to-end)
├── test_textbooks.py        # Smoke test script
├── requirements.txt
├── .env.example
└── README.md
```

## Ablation Modes

| Mode          | Stage 1 (Rewrite) | Stage 3 (Rerank) | Description                |
|---------------|:-----------------:|:----------------:|----------------------------|
| `dual`        | ✅                | ✅               | Full proposal              |
| `input_only`  | ✅                | ❌               | PICO rewriting only        |
| `output_only` | ❌                | ✅               | PICO reranking only        |
| `standard`    | ❌                | ❌               | Vanilla RAG baseline       |
