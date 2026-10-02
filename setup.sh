#!/usr/bin/env bash
# setup.sh — Install all dependencies in the correct order
# Usage: bash setup.sh [--cuda]
#   --cuda : install GPU-accelerated torch (CUDA 12.1)
#   (no flag) : install CPU-only torch

set -e

echo "========================================"
echo "  Dual-PICO RAG — Environment Setup"
echo "========================================"

# Step 1: install torch first (controls numpy ABI)
if [[ "$1" == "--cuda" ]]; then
    echo "[1/4] Installing PyTorch (CUDA 12.1)..."
    pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121
else
    echo "[1/4] Installing PyTorch (CPU)..."
    pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu
fi

# Step 2: pin numpy (must be after torch, before faiss)
echo "[2/4] Installing numpy 1.26.4..."
pip install numpy==1.26.4

# Step 3: remaining requirements
echo "[3/4] Installing remaining packages..."
pip install -r requirements.txt

# Step 4: verify
echo "[4/4] Verifying installation..."
python -c "
import numpy as np
import torch
import faiss
import sentence_transformers
print(f'  numpy      : {np.__version__}')
print(f'  torch      : {torch.__version__}')
print(f'  faiss      : {faiss.__version__}')
print(f'  CUDA avail : {torch.cuda.is_available()}')
print()
print('  Setup OK!')
"

echo "========================================"
echo "  Next: cp .env.example .env"
echo "        # Edit .env with your API key"
echo "        python test_textbooks.py"
echo "========================================"
