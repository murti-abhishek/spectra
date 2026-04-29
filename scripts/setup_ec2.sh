#!/usr/bin/env bash
# Bootstrap script for g5.xlarge (A10G GPU, Ubuntu 22.04, 16 vCPU, 64 GB RAM)
# Run as: bash setup_ec2.sh
set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
REPO_URL="https://github.com/murti-abhishek/spectra.git"
CONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
SPECTRA_DIR="$HOME/spectra"
CACHE_DIR="$HOME/.cache/spectra"
HF_HOME="$HOME/.cache/huggingface"

# ── 1. System deps ────────────────────────────────────────────────────────────
echo "==> Updating system packages"
sudo apt-get update -qq
sudo apt-get install -y -qq git wget curl htop nvtop

# ── 2. Miniconda ─────────────────────────────────────────────────────────────
if ! command -v conda &>/dev/null; then
    echo "==> Installing Miniconda"
    wget -q "$CONDA_URL" -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p "$HOME/miniconda"
    eval "$("$HOME/miniconda/bin/conda" shell.bash hook)"
    conda init bash
    echo "source $HOME/miniconda/etc/profile.d/conda.sh" >> ~/.bashrc
else
    echo "==> conda already installed, skipping"
    eval "$(conda shell.bash hook)"
fi

# ── 3. Clone repo ─────────────────────────────────────────────────────────────
if [ ! -d "$SPECTRA_DIR" ]; then
    echo "==> Cloning spectra"
    git clone "$REPO_URL" "$SPECTRA_DIR"
else
    echo "==> Pulling latest spectra"
    git -C "$SPECTRA_DIR" pull
fi

# ── 4. Create conda environment (CPU base from environment.yml) ───────────────
echo "==> Creating spectra conda environment"
conda env create -f "$SPECTRA_DIR/environment.yml" --force

# ── 5. Install GPU PyTorch (overrides CPU torch if present) ──────────────────
echo "==> Installing GPU PyTorch (CUDA 12.1)"
conda run -n spectra pip install --upgrade \
    torch==2.3.0 torchvision --index-url https://download.pytorch.org/whl/cu121

# ── 6. Install Geneformer deps ────────────────────────────────────────────────
echo "==> Installing transformers + HuggingFace deps"
conda run -n spectra pip install \
    transformers>=4.40 \
    huggingface_hub>=0.23 \
    accelerate>=0.30

# ── 7. Install CELLxGENE Census ───────────────────────────────────────────────
echo "==> Installing cellxgene-census"
conda run -n spectra pip install cellxgene-census

# ── 8. Install scanpy (conda handles llvmlite correctly) ─────────────────────
echo "==> Installing scanpy via conda-forge"
conda install -n spectra -c conda-forge scanpy -y

# ── 9. Configure model weight caching ─────────────────────────────────────────
mkdir -p "$CACHE_DIR" "$HF_HOME"
echo "export HF_HOME=$HF_HOME" >> ~/.bashrc
echo "export SPECTRA_CACHE=$CACHE_DIR" >> ~/.bashrc

# ── 10. Pre-download Geneformer 6L weights ────────────────────────────────────
echo "==> Pre-downloading Geneformer 6L weights"
conda run -n spectra python - <<'EOF'
from huggingface_hub import snapshot_download
snapshot_download("ctheodoris/Geneformer", ignore_patterns=["*.msgpack", "*.h5"])
print("Geneformer weights downloaded.")
EOF

# ── 11. Smoke test ────────────────────────────────────────────────────────────
echo "==> Running smoke test"
conda run -n spectra python - <<'EOF'
import torch
print(f"PyTorch: {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
from spectra.models import MockModelAdapter, PCABaselineAdapter
print("spectra imports OK")
EOF

echo ""
echo "✓ Setup complete. Activate with: conda activate spectra"
echo "  Run benchmark:  python scripts/run_benchmark.py --config configs/dev.yaml"
